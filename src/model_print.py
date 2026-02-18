from models.basic_model import BasicModel

curr = BasicModel(input_shape = (150, 150, 3), categories_count=3)
curr._define_model(input_shape = (150, 150, 3), categories_count=3)

curr.print_summary()