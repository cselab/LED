#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import torch
import torch.nn as nn


def bptta_loss(crnn_model, input_batch, target_batch, latent_states,
               initial_hidden_states, is_train):
    idx_t = crnn_model.parent.sequence_length - crnn_model.parent.iterative_loss_length
    output_batch_iter, last_hidden_state_iter, latent_states_iter, latent_states_pred_iter, RNN_outputs_iter, input_batch_decoded_iter, _ = crnn_model.parent.model.forward(
        input_batch,
        initial_hidden_states,
        is_iterative_forecasting=True,
        horizon=None,
        is_train=is_train,
        teacher_forcing_forecasting=idx_t,
        input_is_latent=False,
        iterative_propagation_is_latent=crnn_model.parent.
        iterative_propagation_is_latent)

    if crnn_model.parent.output_forecasting_loss:
        loss_iter, _ = crnn_model.parent.getLoss(output_batch_iter,
                                                 target_batch)
    else:
        loss_iter = crnn_model.parent.torchZero()

    if crnn_model.parent.latent_forecasting_loss:
        loss_dyn_iter, _ = crnn_model.parent.getLoss(
            latent_states_pred_iter[:, :-1, :],
            latent_states[:, 1:, :],
            is_latent=True)

    else:
        loss_dyn_iter = crnn_model.parent.torchZero()

    # if crnn_model.parent.reconstruction_loss:
    #     loss_auto_iter, _ = crnn_model.parent.getLoss(input_batch_decoded_iter,
    #                                                   input_batch)
    # else:
    loss_auto_iter = crnn_model.parent.torchZero()
    return loss_iter, loss_dyn_iter, loss_auto_iter
