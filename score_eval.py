import tqdm
import json
import pandas as pd
from pycocotools.coco import COCO
from torchvision.datasets import CocoCaptions
from pycocoevalcap.eval import COCOEvalCap

coco = COCO("./data/annotations/captions_val2014.json")
res_file = "./results/captions_model.json"
out_file = "./results/val2014_scores.xlsx"

# evaluate best captions against gt
coco_result = coco.loadRes(res_file)
cocoEval = COCOEvalCap(coco, coco_result)
cocoEval.params['image_id'] = coco_result.getImgIds()
cocoEval.evaluate()

indices = ["BLEU 1-gram", "BLEU 2-gram", "BLEU 3-gram", "BLEU 4-gram",
           "METEOR", "ROUGE_L", "CIDEr"]
data = [cocoEval.eval['Bleu_1']] + [cocoEval.eval['Bleu_2']] + [cocoEval.eval['Bleu_3']] + [cocoEval.eval['Bleu_4']] + \
       [cocoEval.eval['METEOR']] + [cocoEval.eval['ROUGE_L']] + [cocoEval.eval['CIDEr']]
results = pd.DataFrame(columns=[f"3 epochs, lr=0.001"], index=indices, data=data)
results.to_excel(out_file)
print(f"Results saved to {out_file}")