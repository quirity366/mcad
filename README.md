# Dependency

We follow [VAR](https://github.com/FoundationVision/VAR) and [CAR](https://github.com/MiracleDance/CAR). 

We are deeply grateful for the excellent contributions of them.

# Download

Download the [vae](https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth), [mcad](https://pan.baidu.com/s/1e0VWIDt8M5Yii1UoNq4atw?pwd=uzjs) and [LongCLIP](https://huggingface.co/BeichenZhang/LongCLIP-B/blob/main/longclip-B.pt) checkpoints.

Modify the path in `config.py`.

# Demo

Run `condition.py` to generate condition image. Then run `demo.py` to generate the defect image.

```cmd
$ python condition.py
$ python demo.py
```
The result will be saved as `result.png`.