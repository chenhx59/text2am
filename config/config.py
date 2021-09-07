model_config_adjust_list = ['vocab_size']

model_config = {
    'embd_hid': 128, 
    'hidden': 128,
    'dropout': 0.1,
    'encoder_num': 2,
    'encoder_name': 'gru',
    'embedding_from_wv': True,
    'embedding_freeze': True
}


training_config_adjust_list = ['device']
training_config = {
    'pad_id': 0,
    'batch_size': 32, 
    'lr': 1e-3,
    'epoch': 20, 
    'log_steps': 500, 
    'prediction_out_file': 'tmp/prediction.pkl'

}


args_type = {
    'block1': 'type0',     
    'block2': 'type0',      
    'block3': 'type0',     
    'block4': 'type0',     
    'block5': 'type0',     
    'robot': 'type1'
}