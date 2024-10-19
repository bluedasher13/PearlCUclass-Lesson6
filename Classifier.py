from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")

results = classifier("今天是個好日子，我們一起去公園散步吧！天氣真好。")

print(results)