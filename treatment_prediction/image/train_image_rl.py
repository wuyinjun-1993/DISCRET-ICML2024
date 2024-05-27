import pandas as pd

import os,sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from treatment_prediction.image.image_data_utils.simulation import run_simulation
from models.Qmod import *
from parse_args_image import parse_args
from image_data_utils.image_dataset import *
import synthetic_lang
from treatment_prediction.create_language import Language
from utils_treatment import load_configs
from treatment_prediction.nlp.nlp_data_utils.featurization import convert_text_to_features
import torch
from baseline_main import classical_baseline_main, classical_baseline_ls

if __name__ == "__main__":
    args = parse_args()
    root_dir = os.path.dirname(os.path.realpath(__file__))
    random_seed = args.seed
    random.seed(random_seed)

    # Set a random seed for NumPy
    np.random.seed(random_seed)

    # Set a random seed for PyTorch
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # DROP_FEATS=None
    # args.data_path = os.path.join(args.data_folder, args.dataset_name)
    # if args.dataset_name == "music":
    #     raw_df = pd.read_csv(os.path.join(args.data_path, 'music.csv'))
    #     df, offset =run_simulation(raw_df, propensities=[0.8, 0.6], 
    #                                 beta_t=1.0, 
    #                                 beta_c=50.0,
    #                                 gamma=1.0   ,
    #                                 cts=True)
    #     id_attr = "index"
    #     outcome_attr = "Y"
    #     count_outcome_attr = "count_Y"
    #     treatment_attr = "T"
    #     text_attr = "text"
    #     DROP_FEATS=['C','Unnamed: 0', 'Unnamed: 0.1']
    #     split_ids = None
    # elif args.dataset_name == "EEEC":
        
        
        
    #     train_df = pd.read_csv(os.path.join(args.data_path, args.treatment_opt.lower()+'_train.csv'))
    #     valid_df = pd.read_csv(os.path.join(args.data_path, args.treatment_opt.lower()+'_dev.csv'))
    #     test_df = pd.read_csv(os.path.join(args.data_path, args.treatment_opt.lower()+'_test.csv'))
        
    #     train_df.reset_index(inplace=True)
    #     valid_df.reset_index(inplace=True)
    #     test_df.reset_index(inplace=True)
        
    #     df = pd.concat([train_df, valid_df, test_df])
    #     train_ids = list(range(len(train_df)))
    #     valid_ids = list(range(len(train_df), len(train_df)+len(valid_df)))
    #     test_ids = list(range(len(train_df)+len(valid_df), len(train_df)+len(valid_df)+len(test_df)))
    #     split_ids = [train_ids, valid_ids, test_ids]
    #     id_attr = "index"
    #     outcome_attr = "POMS_label"
    #     count_outcome_attr = None
    #     treatment_attr = args.treatment_opt + "_F_label"
    #     text_attr = "Sentence_F"
    #     DROP_FEATS=['ID_F', 'ID_CF', 'Person_F', 'Person_CF', 'Sentence_CF', args.treatment_opt + '_CF_label']
    
    
    # df = convert_text_to_features(args, df, text_attr, treatment_attr, outcome_attr)
    
    # train_df, valid_df, test_df = split_train_valid_test_df(df, split_ids=split_ids)
    
    # # data, id_attr, outcome_attr, count_outcome_attr, treatment_attr, does_attr, text_attr, precomputed, lang, num_feats=None
    # synthetic_lang.DROP_FEATS=DROP_FEATS
    # lang = Language(df, id_attr, outcome_attr, count_outcome_attr, treatment_attr, None, text_attr, precomputed=None, lang=synthetic_lang)
       
    # train_dataset, valid_dataset, test_dataset, feat_range_mappings = create_dataset(df, train_df, valid_df, test_df, synthetic_lang, id_attr, outcome_attr, treatment_attr, text_attr, synthetic_lang.DROP_FEATS, count_outcome_attr=count_outcome_attr)
    # lang = set_lang_data(lang, train_dataset)
    
    # train_dataset, valid_dataset, test_dataset, feat_range_mappings, id_attr, outcome_attr, count_outcome_attr, treatment_attr, lang
    train_dataset, valid_dataset, test_dataset, feat_range_mappings, id_attr, outcome_attr, count_outcome_attr, treatment_attr, lang = construct_dataset_main(args)
    
    args.cont_treatment = False
    args.has_dose = False
    args.num_treatments = 2
    
    numeric_count  = len(train_dataset.num_cols)
        # numeric_count  = len(train_dataset.num_cols) if "num_feat" in lang.syntax else 0
    category_count = list(train_dataset.cat_cols) #len(lang.syntax["cat_feat"]) if "cat_feat" in lang.syntax else 0
    category_sum_count = train_dataset.cat_sum_count
    
    if not args.method == "ours":
        if args.method in classical_baseline_ls:
            classical_baseline_main(args, args.method, train_dataset.transformed_features, train_dataset.treatment_array,train_dataset.dose_array, train_dataset.outcome_array,
                                valid_dataset.transformed_features, valid_dataset.treatment_array, valid_dataset.dose_array, valid_dataset.outcome_array, 
                                test_dataset.transformed_features, test_dataset.treatment_array, test_dataset.dose_array, test_dataset.outcome_array, 
                                 train_dataset, test_dataset, train_dataset.count_outcome_array, valid_dataset.count_outcome_array, test_dataset.count_outcome_array)
        exit(1)
    rl_config, model_config = load_configs(args, root_dir=root_dir)
    # torch.autograd.set_detect_anomaly(True)
    # id_attr, outcome_attr, treatment_attr, lang, learning_rate, gamma, dropout_p, feat_range_mappings, program_max_len, replay_memory_capacity, rl_config, model_config, numeric_count, category_count, category_sum_count, args, topk_act=1, num_labels=2, a_weight=1.0, y_weight=1.0, mlm_weight=1.0,
    mod = QNet_rl(train_dataset, valid_dataset, test_dataset, id_attr, outcome_attr, treatment_attr, lang, args.lr, rl_config["gamma"], 
            args.dropout_p, feat_range_mappings, args.program_max_len,
            rl_config["replay_memory_capacity"], rl_config, model_config, numeric_count, category_count, category_sum_count, args, topk_act=args.topk_act, num_labels=2, 
            batch_size = args.batch_size, # batch size for training
            a_weight = 0.1,  # loss weight for A ~ text
            y_weight = 0.1,  # loss weight for Y ~ A + textq
            mlm_weight=1.0,  # loss weight for DistlBert
            modeldir=args.log_folder, log_folder = args.log_folder, classification = False) # directory for saving the best model

    # if args.cached_model_name is not None:
    #     mod.dqn.policy_net.load_state_dict(torch.load(args.cached_model_name, map_location=mod.device))
    #     mod.dqn.target_net.load_state_dict(torch.load(args.cached_model_name, map_location=mod.device))
    # load_pretrained_backbone_models_rl(mod, os.path.join(args.log_folder, "_bestmod.pt"))
    # mod.dqn.policy_net.load_state_dict(torch.load(os.path.join(args.log_folder, "policy_net")))
    # mod.dqn.target_net.load_state_dict(torch.load(os.path.join(args.log_folder, "target_net")))
    mod.train(train_dataset, valid_dataset, test_dataset)#,  # texts in training data)
    
    # mod.train(df['text'],  # texts in training data``
    #         df['T'],     # treatments in training data
    #         df['C'],     # confounds in training data, binary
    #         df['Y'],     # outcomes in training data
    #         epochs=20,   # the maximum number of training epochs
    #         learning_rate = 2e-5)  # learning rate for the training
