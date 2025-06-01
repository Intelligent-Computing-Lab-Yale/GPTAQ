import utils
import torch
import model_utils
import data_utils
import eval_utils
import quant_utils
import gptq_utils
import gptaq_utils


def configure_act_quantizer(model, args):
    qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
    down_proj_groupsize = -1
    if args.a_groupsize > 0 and "llama" in args.model:
        down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)

    for name in qlayers:
        layer_input_bits = args.a_bits
        layer_groupsize = args.a_groupsize
        layer_a_sym = not (args.a_asym)
        layer_a_clip = args.a_clip_ratio

        if 'head' in name:  # Skip lm_head quantization
            layer_input_bits = 16

        if 'down_proj' in name:  # Set the down_proj precision
            if args.int8_down_proj:
                layer_input_bits = 8
            layer_groupsize = down_proj_groupsize

        qlayers[name].quantizer.configure(bits=layer_input_bits,
                                          groupsize=layer_groupsize,
                                          sym=layer_a_sym,
                                          clip_ratio=layer_a_clip)


def main():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)

    torch.manual_seed(args.seed)
    model = model_utils.get_model(args.model)
    if args.eval_fp:
        eval_utils.test(args, model)
    quant_utils.add_actquant(model)  # Add Activation Wrapper to the model as the rest of the code assumes it is present

    # Add Input Quantization
    if args.a_bits < 16 and args.enable_aq_calibration:
        configure_act_quantizer(model, args)

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:  # GPTQ Weight Quantization

            calib_data = data_utils.get_calibration_data(
                args.cal_dataset, num_data=args.nsamples, model=model,
            )

            if args.asym_calibrate:
                quantizers = gptaq_utils.gptaq_fwrd(model, calib_data, utils.DEV, args)
                save_dict["w_quantizers"] = quantizers
            else:
                quantizers = gptq_utils.gptq_fwrd(model, calib_data, utils.DEV, args)
                save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

    if args.a_bits < 16 and not args.enable_aq_calibration:
        configure_act_quantizer(model, args)

    # Evaluating on dataset
    eval_utils.test(args, model)


if __name__ == '__main__':
    main()
