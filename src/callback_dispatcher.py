from callbacks import MoveToGPUCallback, TrackResultLM, TrackResultSentimentClf

callbacks = {
    "rnnsentimentclf": [MoveToGPUCallback(), TrackResultSentimentClf()],
    "grusentimentclf": [MoveToGPUCallback(), TrackResultSentimentClf()],
    "lstmsentimentclf": [MoveToGPUCallback(), TrackResultSentimentClf()],
    "rnnlanguagemodel": [MoveToGPUCallback(), TrackResultLM()],
    "grulanguagemodel": [MoveToGPUCallback(), TrackResultLM()],
    "lstmlanguagemodel": [MoveToGPUCallback(), TrackResultLM()],
    'test': [MoveToGPUCallback(), TrackResultSentimentClf()]
}
