def merge_predictions(predictions):
    if not predictions:
        return []
    merged = []
    current = dict(predictions[0])
    for pred in predictions[1:]:
        if (pred['start'] == current['end'] and pred['entity'] == current['entity'] and pred['entity'] != 'O'):
            current['end'] = pred['end']
        else:
            merged.append(current)
            current = dict(pred)
    merged.append(current)
    return merged