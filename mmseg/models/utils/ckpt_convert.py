# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from collections import OrderedDict

import torch


def mit_convert(ckpt):
    new_ckpt = OrderedDict()
    # Process the concat between q linear weights and kv linear weights
    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('patch_embed'):
            stage_i = int(k.split('.')[0].replace('patch_embed', ''))
            new_k = k.replace(f'patch_embed{stage_i}', f'layers.{stage_i-1}.0')
            new_v = v
            if 'proj.' in new_k:
                new_k = new_k.replace('proj.', 'projection.')
        elif k.startswith('block'):
            stage_i = int(k.split('.')[0].replace('block', ''))
            new_k = k.replace(f'block{stage_i}', f'layers.{stage_i-1}.1')
            new_v = v
            if 'attn.q.' in new_k:
                sub_item_k = k.replace('q.', 'kv.')
                new_k = new_k.replace('q.', 'attn.in_proj_')
                new_v = torch.cat([v, ckpt[sub_item_k]], dim=0)
            elif 'attn.kv.' in new_k:
                continue
            elif 'attn.proj.' in new_k:
                new_k = new_k.replace('proj.', 'attn.out_proj.')
            elif 'attn.sr.' in new_k:
                new_k = new_k.replace('sr.', 'sr.')
            elif 'mlp.' in new_k:
                string = f'{new_k}-'
                new_k = new_k.replace('mlp.', 'ffn.layers.')
                if 'fc1.weight' in new_k or 'fc2.weight' in new_k:
                    new_v = v.reshape((*v.shape, 1, 1))
                new_k = new_k.replace('fc1.', '0.')
                new_k = new_k.replace('dwconv.dwconv.', '1.')
                new_k = new_k.replace('fc2.', '4.')
                string += f'{new_k} {v.shape}-{new_v.shape}'
                # print(string)
        elif k.startswith('norm'):
            stage_i = int(k.split('.')[0].replace('norm', ''))
            new_k = k.replace(f'norm{stage_i}', f'layers.{stage_i-1}.2')
            new_v = v
        else:
            new_k = k
            new_v = v
        new_ckpt[new_k] = new_v
    return new_ckpt
