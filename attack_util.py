import copy
from sapag import sapag_attack
from invgrad import invgrad_attack
from cpl import cpl_attack
from dlg import dlg_attack
from idlg import idlg_attack
from stgia import stgia_attack


def attack(attack_method,
           attack_iteration,
           true_dy_dx, gt_data_np, gt_tim_np, gt_label_np,
           save_path, er,
           original_out, args, model, device, dataset,
           last_attack_record,
           is_use_last_result,
           is_use_road
           ):
    attack_record = None
    if attack_method == 'dlg':
        attack_record = dlg_attack(args,
                                   batch_size=args.batch_size,
                                   model=copy.deepcopy(model),
                                   true_dy_dx=true_dy_dx,
                                   dlg_attack_round=args.dlg_attack_rounds,
                                   dlg_iteration=args.dlg_iterations,
                                   dlg_lr=args.dlg_lr,
                                   global_iter=attack_iteration,
                                   model_name=args.model_name,
                                   gt_data_np=gt_data_np,
                                   gt_tim_np=gt_tim_np,
                                   gt_label_np=gt_label_np,
                                   save_path=save_path,
                                   device=device,
                                   dataset=dataset,
                                   er=er,
                                   )
    elif attack_method == 'idlg':
        attack_record = idlg_attack(args,
                                    batch_size=args.batch_size,
                                    model=copy.deepcopy(model),
                                    true_dy_dx=true_dy_dx,
                                    dlg_attack_round=args.dlg_attack_rounds,
                                    dlg_iteration=args.dlg_iterations,
                                    dlg_lr=args.dlg_lr,
                                    global_iter=attack_iteration,
                                    model_name=args.model_name,
                                    gt_data_np=gt_data_np,
                                    gt_tim_np=gt_tim_np,
                                    gt_label_np=gt_label_np,
                                    save_path=save_path,
                                    device=device,
                                    dataset=dataset,
                                    er=er,
                                    )
    elif attack_method == 'cpl':
        attack_record = cpl_attack(args,
                                   batch_size=args.batch_size,
                                   model=copy.deepcopy(model),
                                   true_dy_dx=true_dy_dx,
                                   dlg_attack_round=args.dlg_attack_rounds,
                                   dlg_iteration=args.dlg_iterations,
                                   dlg_lr=args.dlg_lr,
                                   global_iter=attack_iteration,
                                   model_name=args.model_name,
                                   gt_data_np=gt_data_np,
                                   gt_tim_np=gt_tim_np,
                                   gt_label_np=gt_label_np,
                                   save_path=save_path,
                                   device=device,
                                   dataset=dataset,
                                   er=er,
                                   original_out=original_out.detach().clone()
                                   )
    elif attack_method == 'sapag':
        attack_record = sapag_attack(args,
                                           batch_size=args.batch_size,
                                           model=copy.deepcopy(model),
                                           true_dy_dx=true_dy_dx,
                                           dlg_attack_round=args.dlg_attack_rounds,
                                           dlg_iteration=args.dlg_iterations,
                                           dlg_lr=args.dlg_lr,
                                           global_iter=attack_iteration,
                                           model_name=args.model_name,
                                           gt_data_np=gt_data_np,
                                           gt_tim_np=gt_tim_np,
                                           gt_label_np=gt_label_np,
                                           save_path=save_path,
                                           device=device,
                                           dataset=dataset,
                                           er=er,
                                           )
    elif attack_method == 'invgrad':
        attack_record = invgrad_attack(args,
                                               batch_size=args.batch_size,
                                               model=copy.deepcopy(model),
                                               true_dy_dx=true_dy_dx,
                                               dlg_attack_round=args.dlg_attack_rounds,
                                               dlg_iteration=args.dlg_iterations,
                                               dlg_lr=args.dlg_lr,
                                               global_iter=attack_iteration,
                                               model_name=args.model_name,
                                               gt_data_np=gt_data_np,
                                               gt_tim_np=gt_tim_np,
                                               gt_label_np=gt_label_np,
                                               save_path=save_path,
                                               device=device,
                                               dataset=dataset,
                                               er=er,
                                               )
    elif attack_method == 'stgia':
        attack_record = stgia_attack(args,
                                     batch_size=args.batch_size,
                                     model=model,
                                     true_dy_dx=true_dy_dx,
                                     dlg_attack_round=args.dlg_attack_rounds,
                                     dlg_iteration=args.dlg_iterations,
                                     dlg_lr=args.dlg_lr,
                                     global_iter=attack_iteration,
                                     model_name=args.model_name,
                                     gt_data_np=gt_data_np,
                                     gt_tim_np=gt_tim_np,
                                     gt_label_np=gt_label_np,
                                     save_path=save_path,
                                     device=device,
                                     dataset=dataset,
                                     er=er,
                                     last_attack_record=last_attack_record,
                                     is_use_last_result=is_use_last_result,
                                     is_use_road=is_use_road)

    return attack_record