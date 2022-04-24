import torch.nn as nn
import torch.nn.functional as F
from KD_loss import patience_loss
import torch


def msd_loss(s_img_answer_scores, t_img_answer_scores, s_txt_answer_scores, t_txt_answer_scores,
             s_img_txt_answer_scores, t_img_txt_answer_scores, target, alpha, T, task, ATKD=False, if_vid = False):
    if ATKD:
        img_tea_std = torch.std(t_img_answer_scores, dim=-1, keepdim=True)
        img_stu_std = torch.std(s_img_answer_scores, dim=-1, keepdim=True)

        txt_tea_std = torch.std(t_txt_answer_scores, dim=-1, keepdim=True)
        txt_stu_std = torch.std(s_txt_answer_scores, dim=-1, keepdim=True)

        img_txt_tea_std = torch.std(t_img_txt_answer_scores, dim=-1, keepdim=True)
        img_txt_stu_std = torch.std(s_img_txt_answer_scores, dim=-1, keepdim=True)

        TM_s = img_stu_std
        TM_t = img_tea_std

        TT_s = txt_stu_std
        TT_t = txt_tea_std

        TTM_s = img_txt_stu_std
        TTM_t = img_txt_tea_std
    else:
        TM_s = TM_t = TT_s = TT_t = TTM_s = TTM_t = T

    img_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(s_img_answer_scores / TM_s, dim=1),
                                              F.softmax(t_img_answer_scores / TM_t,
                                                        dim=1))
    txt_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(s_txt_answer_scores / TT_s, dim=1),
                                              F.softmax(t_txt_answer_scores / TT_t,
                                                        dim=1))
    img_txt_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(s_img_txt_answer_scores / TTM_s, dim=1),
                                                  F.softmax(t_img_txt_answer_scores / TTM_t,
                                                            dim=1))
    # print("s_img_answer_scores.shape", s_img_answer_scores.shape)
    # print("s_img_answer_scores", s_img_answer_scores)
    # print("target.shape", target.shape)
    # print("target", target)
    # # print(torch.std(s_img_answer_scores, dim=-1, keepdim=True))
    # # print(torch.std(s_img_answer_scores, dim=1, keepdim=True))
    # print("img_tea_std", img_tea_std)
    # # print("img_txt_tea_std", img_txt_tea_std)
    # # print("img_loss",img_loss)
    # # print("img_txt_loss", img_txt_loss)
    # exit()

    if task == 'vqa' or task == 've':
        # vqa,ve loss
        img_loss = img_loss.mean()
        txt_loss = txt_loss.mean()
        img_txt_loss = img_txt_loss.mean()
        nll_loss = F.binary_cross_entropy_with_logits(s_img_txt_answer_scores, target, reduction='none')
        nll_loss = nll_loss.mean()

    if task == 'nlvr2':
        # nlvr2 loss
        img_loss = img_loss.sum(1).mean()
        txt_loss = txt_loss.sum(1).mean()
        img_txt_loss = img_txt_loss.mean()
        nll_loss = F.cross_entropy(
            s_img_txt_answer_scores, target, reduction='none')
        nll_loss = nll_loss.mean()

    if task == 'vcr':
        img_loss = img_loss.mean()
        txt_loss = txt_loss.mean()
        img_txt_loss = img_txt_loss.mean()
        nll_loss = F.cross_entropy(
            s_img_txt_answer_scores, target.squeeze(-1),
            reduction='mean')
    if if_vid:
        vid_loss = patience_loss(teacher_patience=t_img_txt_answer_scores, student_patience=s_img_txt_answer_scores,
                              normalized_patience=True, mi='vid').mean()
        tol_loss = ((1.0 - alpha) * (0.25 * img_loss + 0.25 * txt_loss + 0.5 * img_txt_loss) + alpha * nll_loss + vid_loss).half()
    else:
        tol_loss = ((1.0 - alpha) * (0.25 * img_loss + 0.25 * txt_loss + 0.5 * img_txt_loss) + alpha * nll_loss).half()

    # print("img_loss", img_loss)
    # print("img_txt_loss", img_txt_loss)
    # print("nll_loss", nll_loss)
    # print("tol_loss", tol_loss)
    # print(torch.sum(torch.sum(F.kl_div(F.log_softmax(s_img_answer_scores / 10, dim=1),
    #                                           F.softmax(t_img_answer_scores / 10,
    #                                                     dim=1)), dim=-1)* (9 * torch.ones(s_img_answer_scores.shape[0],1).cuda())) /s_img_answer_scores.shape[0]/ s_img_answer_scores.shape[0])


    return tol_loss


def get_p_sum(t_encoder_layers, s_encoder_layers, mi, p_list=None):
    teacher_patience = []
    student_patience = s_encoder_layers
    p_loss = []

    if p_list:
        for index, p_sample in enumerate(p_list):
            teacher_patience.append(t_encoder_layers[index * 2 + p_list[index]])
    else:
        if (len(s_encoder_layers) == 6):
            if len(t_encoder_layers) == 24:
                for i in range(24):
                    if i % 4 == 0:
                        teacher_patience.append(t_encoder_layers[i])
            else:
                for i in range(12):
                    if i % 2 == 0:
                        teacher_patience.append(t_encoder_layers[i])
        elif (len(s_encoder_layers) == 3):
            teacher_patience.append(t_encoder_layers[0])
            teacher_patience.append(t_encoder_layers[5])
            teacher_patience.append(t_encoder_layers[11])
        else:
            teacher_patience = t_encoder_layers

    p_loss = []
    for j in range(len(teacher_patience)):
        p_loss.append(
            patience_loss(teacher_patience=teacher_patience[j], student_patience=student_patience[j],
                          normalized_patience=True, mi=mi).mean())
    # p_sum = torch.zeros(1, dtype=torch.float32).cuda()
    p_sum = 0
    gama = 0.8
    last = (1 - gama) / (len(p_loss) - 1)
    for i in range(len(p_loss)):
        if i == len(p_loss):
            p_sum = p_sum + p_loss[i] * gama
            # print(p_loss[i])
        else:
            p_sum = p_sum + p_loss[i] * last
    return p_sum.half()