from train import *

def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
        
    for vis_backend in cfg.visualizer.vis_backends:
           if vis_backend.type == "WandbVisBackend":
               # Name the job type after config file name.
               job_type = osp.splitext(osp.basename(args.config))[0]
               # Name the job after the last part of the work directory.
               job_name = cfg.work_dir.split("/")[-1]
               # Hash job name into a 8-digit id.
               job_id = hashlib.sha256(job_name.encode()).hexdigest()[:8]
               vis_backend.init_kwargs.job_type = job_type
               vis_backend.init_kwargs.name = job_name
               vis_backend.init_kwargs.id = job_id
               vis_backend.init_kwargs.resume = "allow"
               vis_backend.init_kwargs.allow_val_change = True
               
        # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == "AmpOptimWrapper":
            print_log(
                "AMP training is already enabled in your config.",
                logger="current",
                level=logging.WARNING,
            )
        else:
            assert optim_wrapper == "OptimWrapper", (
                "`--amp` is only supported when the optimizer wrapper type is "
                f"`OptimWrapper` but got {optim_wrapper}."
            )
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)