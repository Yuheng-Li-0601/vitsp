# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union
import random
import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

import torch.utils.checkpoint as cp

class Gating(nn.Module):
    def __init__(self, hidden_size=2048, expansion_factor=4, dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        mid_dim = hidden_size * expansion_factor

        def mlp_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, in_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(in_dim),
            )

        self.block1 = mlp_block(hidden_size, mid_dim)
        self.block2 = mlp_block(hidden_size, mid_dim)
        self.block3 = mlp_block(hidden_size, mid_dim)
        self.block4 = mlp_block(hidden_size, mid_dim)

        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2)  # 2 experts
        )

    def forward(self, x):
        if self.use_checkpoint:
            x = x + cp.checkpoint(self.block1, x)
            x = x + cp.checkpoint(self.block2, x)
            x = x + cp.checkpoint(self.block3, x)
            x = x + cp.checkpoint(self.block4, x)
        else:
            x = x + self.block1(x)
            x = x + self.block2(x)
            x = x + self.block3(x)
            x = x + self.block4(x)

        logits = self.gate(x)  # shape: [B, 2]
        probs = torch.softmax(logits, dim=-1)  # 每个 token 的 expert 选择概率
        return probs
        

class CrossAttentionPooling(nn.Module):
    def __init__(self, dim, num_heads=16):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, dim))  # [1, D]
        
        self.attn1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        self.attn2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.attn3 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)

        self.attn4 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, batched_tokens: list[torch.Tensor]):
        """
        batched_tokens: List of Tensors of shape [Ti, D], length = B
        """
        B = len(batched_tokens)
        D = batched_tokens[0].shape[-1]
        device = batched_tokens[0].device

        # 1. Padding
        max_len = max(t.shape[0] for t in batched_tokens)
        dtype = self.query_token.dtype
        padded = torch.zeros(B, max_len, D, dtype=dtype, device=device)
        padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)

        for i, t in enumerate(batched_tokens):
            L = t.shape[0]
            padded[i, :L] = t
            padding_mask[i, :L] = False

        # 2. Query token: [B, 1, D]
        query = self.query_token.unsqueeze(0).expand(B, -1, -1)  # learnable token for each sample

        # 3. First attention
        out1, _ = self.attn1(query, padded, padded, key_padding_mask=padding_mask)  # [B, 1, D]
        out1 = self.norm1(out1)

        # 4. Second attention
        out2, _ = self.attn2(out1, padded, padded, key_padding_mask=padding_mask)  # [B, 1, D]
        out2 = self.norm2(out2)

        out3, _ = self.attn2(out2, padded, padded, key_padding_mask=padding_mask)  # [B, 1, D]
        out3 = self.norm2(out3)

        out4, _ = self.attn2(out3, padded, padded, key_padding_mask=padding_mask)  # [B, 1, D]
        out4 = self.norm2(out4)

        return out4.squeeze(1)
    
class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "InternVisionModel",
        "Qwen3DecoderLayer",
    ]

    # support transformers 4.51.+
    _tp_plan = ''

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            architecture: str = config.llm_config.architectures[0]
            if architecture == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif architecture == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            elif architecture == 'Qwen3MoeForCausalLM':
                self.language_model = Qwen3MoeForCausalLM(config.llm_config)
            elif architecture == 'Qwen3ForCausalLM':
                self.language_model = Qwen3ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{architecture} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        
        self.mlp2 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 4),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 4, llm_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llm_hidden_size * 2, llm_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llm_hidden_size * 2, llm_hidden_size)
        )

        self.pooling_before_gating = CrossAttentionPooling(dim=vit_hidden_size)
        self.gating = Gating(hidden_size=vit_hidden_size)
        
        self.flash_mode = getattr(config, "flash_mode", False)
        if self.flash_mode:
            self.flash_relative_threshold = config.flash_relative_threshold
            self.flash_absolute_threshold = config.flash_absolute_threshold

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message


    def compress_visual_tokens_in_sentence(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        mask_idx: torch.Tensor,
        img_context_token_id: int,
        gate_result,
    ) -> tuple:
        
        N, C = input_embeds.shape
        
        input_ids = input_ids.squeeze(0)  # (N,)
        selected = (input_ids == img_context_token_id)
        padded = torch.cat([torch.tensor([0], device=selected.device), selected.int(), torch.tensor([0], device=selected.device)])
        diff = torch.diff(padded)

        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends = (diff == -1).nonzero(as_tuple=True)[0]
        lengths = ends - starts

        keep_mask = torch.ones(N, dtype=torch.bool, device=input_embeds.device)

        delete_flags = torch.zeros(N, dtype=torch.int32, device=input_embeds.device)

        p = random.uniform(0, 1)

        total_blocks = 0
        block_counts = []
        for l in lengths.tolist():
            if l % 256 != 0:
                raise ValueError(f"l % 256 != 0, l = {l}")
            num_blocks = l // 256
            block_counts.append(num_blocks)
            total_blocks += num_blocks

        flag_idx = 0
        for s, e, l, num_blocks in zip(starts.tolist(), ends.tolist(), lengths.tolist(), block_counts):
            for i in range(num_blocks):
                block_start = s + i * 256
                block_end = block_start + 256

                compress = gate_result[flag_idx]
                flag_idx += 1

                if compress:
                    keep_mask[block_start + 64 : block_end] = False
                    delete_flags[block_start + 64 : block_end] = 1

        cumulative_deletes = torch.cumsum(delete_flags, dim=0)
        cumulative_deletes = torch.cat([cumulative_deletes,  cumulative_deletes[-1:].clone()], dim=0)
        
        
        mask_idx = mask_idx.squeeze(0)
        updated_mask_idx = mask_idx - cumulative_deletes[mask_idx.to(cumulative_deletes.device)].to(mask_idx.device)
        updated_mask_idx = updated_mask_idx.unsqueeze(0)

        new_input_embeds = input_embeds[keep_mask.to(input_embeds.device), :]
        new_input_ids = input_ids[keep_mask.to(input_ids.device)]

        return new_input_embeds, new_input_ids, updated_mask_idx, keep_mask
    
    def get_image_num_per_sample(
            self,
            input_ids: torch.Tensor,
    ):
        input_ids = input_ids.squeeze(0)  # (N,)
        selected = (input_ids == self.img_context_token_id)
        padded = torch.cat([torch.tensor([0], device=selected.device), selected.int(), torch.tensor([0], device=selected.device)])
        diff = torch.diff(padded)

        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends = (diff == -1).nonzero(as_tuple=True)[0]
        lengths = ends - starts

        return lengths
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def split_and_merge(self, features: torch.Tensor, split_sizes: torch.Tensor):
        """
        features: Tensor of shape [T, 1024, 1024]
        split_sizes: 1D Tensor like [3, 3, 4] — 每个样本 tile 数

        returns: List of Tensors of shape [tile_i * 1024, 1024]
        """
        # 拆分 features → 每个样本一个 tile list
        tile_splits = torch.split(features, split_sizes, dim=0)

        # 合并前两维：tile * 1024 × 1024
        merged = [x.reshape(-1, x.shape[-1]) for x in tile_splits]

        return merged
        
    def extract_feature_flash(self, pixel_values, lengths):

        with torch.no_grad():
            vit_embeds_1024 = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state

        vit_embeds_1024 = vit_embeds_1024[:, 1:, :]
        h = w = int(vit_embeds_1024.shape[1] ** 0.5)
        vit_embeds_1024 = vit_embeds_1024.reshape(vit_embeds_1024.shape[0], h, w, -1)       

        # begin moe
        lengths = [int(x) for x in lengths.tolist()]
        vit_embeds_1024_split_and_merge = self.split_and_merge(vit_embeds_1024, lengths)

        gate = self.pooling_before_gating(vit_embeds_1024_split_and_merge)
        gate = self.gating(gate)

        vit_embeds_256 = vit_embeds_1024.clone()

        with torch.no_grad():
            vit_embeds_64 = self.pixel_shuffle(vit_embeds_1024, scale_factor=self.downsample_ratio ** 2)
            vit_embeds_64 = vit_embeds_64.reshape(vit_embeds_64.shape[0], -1, vit_embeds_64.shape[-1])
            vit_embeds_64 = self.mlp2(vit_embeds_64)

            vit_embeds_256 = self.pixel_shuffle(vit_embeds_256, scale_factor=self.downsample_ratio)
            vit_embeds_256= vit_embeds_256.reshape(vit_embeds_256.shape[0], -1, vit_embeds_256.shape[-1])
            vit_embeds_256 = self.mlp1(vit_embeds_256)

        return vit_embeds_64, vit_embeds_256, gate

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate_flash(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                lengths = self.get_image_num_per_sample(input_ids) / 256

                lengths_sum = torch.ones(int(lengths.sum().item()), dtype=torch.int64)
                lengths = lengths_sum.repeat_interleave(1) 
                vit_embeds_64, vit_embeds_256, gate_result = self.extract_feature_flash(pixel_values, lengths)
            
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
    
            input_ids = input_ids.reshape(B * N)

            relative_threshold_value = torch.quantile(gate_result[:, 0].to(torch.float32), self.flash_relative_threshold)
            gate_result = (gate_result[:, 0] > relative_threshold_value) & (gate_result[:, 0] >= self.flash_absolute_threshold)

            selected_embeds = []
            for i in range(gate_result.size(0)):
                if gate_result [i]:
                    selected_embeds.append(vit_embeds_64[i]) 
                else:
                    selected_embeds.append(vit_embeds_256[i])

            vit_embeds = torch.cat(selected_embeds, dim=0)

            assert torch.all(attention_mask == 1)
            input_embeds, input_ids, attention_mask, keep_mask = self.compress_visual_tokens_in_sentence(
                input_embeds=input_embeds,
                input_ids=input_ids,
                mask_idx=attention_mask,
                img_context_token_id=self.img_context_token_id,
                gate_result=gate_result,
            )
            
            attention_mask = torch.ones(1, input_embeds.shape[0]).to(input_embeds.device)

            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, -1, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @torch.no_grad()
    def generate_normal(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        
        if getattr(self, "flash_mode", False):
            return self.generate_flash(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_features,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                **generate_kwargs,
            )
        else:
            return self.generate_normal(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_features,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                **generate_kwargs,
            )
        
    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
