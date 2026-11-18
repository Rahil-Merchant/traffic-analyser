"""Download a COCO-pretrained DEIMv2 checkpoint from Hugging Face and save it in the layout DEIMv2's
`train.py -t` expects ({'model': state_dict}).

    python training/fetch_deimv2_ckpt.py --size l --out <DEIMv2 repo>/ckpts/deimv2_dinov3_l_coco.pth

The Hugging Face repos (Intellindust/DEIMv2_DINOv3_<SIZE>_COCO) are public and hold the same weights as the
Google Drive links in the DEIMv2 README, saved by a wrapper whose attribute names (backbone, encoder,
decoder) match DEIMv2's model, so the keys load unchanged. The 80-class heads do not match our 9 classes;
the tuning loader skips those tensors and the heads start fresh. Because the checkpoint carries the whole
backbone, Meta's gated DINOv3 weights are not needed when tuning from it.
"""
import argparse
from pathlib import Path

REPOS = {'s': 'DEIMv2_DINOv3_S_COCO', 'm': 'DEIMv2_DINOv3_M_COCO', 'l': 'DEIMv2_DINOv3_L_COCO', 'x': 'DEIMv2_DINOv3_X_COCO'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--size', choices=sorted(REPOS), default='l')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print(f'{out} exists')
        return
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    path = hf_hub_download(f'Intellindust/{REPOS[a.size]}', 'model.safetensors')
    sd = load_file(path)
    prefixes = sorted({k.split('.')[0] for k in sd})
    if prefixes != ['backbone', 'decoder', 'encoder']:
        raise SystemExit(f'unexpected key prefixes {prefixes}; the Hugging Face layout changed')
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': sd}, out)
    print(f'saved {len(sd)} tensors to {out}')


if __name__ == '__main__':
    main()
