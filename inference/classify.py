from transformers import pipeline

PREDICTION_THRESHOLD = 0.45
classifier = pipeline(model="facebook/bart-large-mnli",device='cuda:0')


def predict_item(item,user_tags,classifier=classifier,log=False,top_n = 2, selection='threshold'):
    """
    Returning a list of suggested tags from the tags the user has created. 
    Threshold filtering to select the tags right now.

    Args:
        block (str): str representation of a block in the Resume 
        user_tags (list): list of all tags created by the user
        classifier (pipeline, optional): Classification Model. Defaults to classifier.
        log (bool, optional): Prints the prediction results. Setting up Logging in the future. Defaults to True.
    """
    #ISSUE: if the number of relevant tags is less than the top_n value, then some predictions will be bogus.
    #ISSUE: threshold value needs to be experiemnted with
    #SOLUTION: Find the point where the confidence drops off drastically, and then use that as the threshold.

    assert selection in ['top_n','threshold']
    if len(user_tags) > 0:
        results = classifier(item,candidate_labels = user_tags)
        preds = list(zip(results['labels'], results['scores']))

        if log:
            print("".join([f"{x[0]}: {x[1]}\n" for x in zip(results['labels'], results['scores'])]))

        if selection == 'top_n':
            preds = sorted(preds,key=lambda x: x[1],reverse=True)[:top_n]
        
        elif selection == "threshold":
            preds = [x[0] for x in filter(lambda x: x[1] > PREDICTION_THRESHOLD, preds)]

        return preds
    
    else:
        return None

if __name__== "__main__":
    input_sentence = "I love developing new Computer Vision softwares using AWS, Azure, GCP "
    user_tags = ['computer vision','cloud','baking','runner']
    print(predict_item(input_sentence,user_tags, selection='threshold'))