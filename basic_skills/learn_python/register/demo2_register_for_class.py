"""
reference: 
"""
from class_register import Register

class Registers:
    model = Register('model')
    dataset = Register('dataset') # lambda x: x

class Model:
    pass

# equal to : Model1 = Registers.model('Model1')
@Registers.model
class Model1(Model):
    pass

@Registers.model
class Model2(Model):
    pass

@Registers.model
class Model3(Model):
    pass


@Registers.dataset
class Dataset1:
    pass


print(Registers.model.items())
print(Registers.dataset.items())
print(type(Registers.model))

model1_instance = Model1()
print(model1_instance)