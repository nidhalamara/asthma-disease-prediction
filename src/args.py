class args:
    def __init__(self,epochs,test_size,randon_state,validation_split,data_path):
        self.test_size=test_size
        self.random_state=randon_state
        self.validation_split=validation_split
        self.epochs=epochs
        self.data_path=data_path
        self.model_path=None

