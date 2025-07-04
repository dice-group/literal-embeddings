from pytorch_lightning import Trainer

class KGETrainer(Trainer):
    def __init__(self,
                evaluator=None,
                entity_dataset = None,
                *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.evaluator = evaluator
        self.entity_dataset = entity_dataset
