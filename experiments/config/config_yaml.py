import yaml

dict = {
    'data_params':
        {
            'alpha': 0.5,
            'K': 4,
        }
    ,
    'model_params':
        {
            'l1': 8,
            'l2': 16,
            'l3': 32,
            'feature': 64,
            'activation': 'leakyrelu',
            'kernel': 7,
        },
    'exp_params':
        {
            'lr': 0.001,
            'weight_decay': 0.0,
        },

    'serach_space':
        {

        }

}
file = './test.yaml'
with open(file, 'w') as f:
    yaml.dump(dict, f)
