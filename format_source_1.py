import json
import os
import sys
import numpy as np
#dic = ["middle","high"]
def race2reco(filename,dirc):
    dataset = []
    middle_dataset = []
    high_dataset = []
    filepath = os.path.join(dirc,filename)
    try:
        data  = json.load(open(filepath),encoding='utf-8')
    except:
        print (filepath)
        exit()
    passage = data["article"]
    passage_id = filename.split("_")[1] + '_' + filename.split("_")[2]
    for query_id,query_text in enumerate(data["questions"]):
        options = data["options"][query_id]
        option_text = "|".join(options)
        #answer = options[ord(data["answers"][query_id])-ord("A")]
        answer = ord(data["answers"][query_id])-ord("A")
        _id = "{}_{}".format(passage_id,query_id)
        data_ = {"alternatives":option_text,"passage":passage,"query_id":_id,"answer":answer,"query":query_text}
        
        dataset.append(data_)
        """
        if passage_id.startswith("middle"):
            middle_dataset.append(data_)
        else:
            high_dataset.append(data_)
        """
    return dataset
if __name__ == "__main__":
    dataset=[]
    dataset_middle = []
    dataset_high = []
    dirc = sys.argv[1]
    for filename in os.listdir(dirc):
            d = race2reco(filename,dirc)
            dataset.extend(d)
    np.random.shuffle(dataset)
    #print(dataset[0])
    json.dump(dataset,open(sys.argv[1]+".json","w",encoding='utf-8'),indent=4, ensure_ascii=False)