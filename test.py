from data_juicer.ops.mapper import ImageOCRMapper

mapper = ImageOCRMapper(low_conf_thresh=0.7)  

sample = {
    "images": ["/Users/xinyu/Documents/SQZ/Project/TextAtlasSample/testOCR/testimages/1.jpeg"],
    "__dj__meta__": {},
}

result = mapper.process_single(sample)

print("OCR texts:", result["__dj__meta__"]["image_ocr_tag"][0][0].get("rec_texts"))
print("Difficulty:", result["__dj__meta__"]["image_ocr_difficulty"][0])