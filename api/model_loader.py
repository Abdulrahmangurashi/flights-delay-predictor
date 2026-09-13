from models.register import ModelRegister

class State:
    def __init__(self):
        self.model=None
        self.processor=None
        self.version=None
        self.alias="champion"
        
state = State()
def load_model():
    register=ModelRegister()
    model,preprocessor,version=register.get_production()
    state.model=model
    state.processor=preprocessor
    state.version=version
    
def get_state():
    return state