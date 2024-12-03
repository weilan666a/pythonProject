export BKCL_PCIE_RING=1
export BKCL_RING_BUFFER_SIZE=8388608
export XPU_PADDLE_L3_SIZE=62914560
export FLAGS_fuse_parameter_memory_size=32
export FLAGS_fuse_parameter_groups_size=32
export XPU_BLACK_LIST=reduce_sum,reduce_max,expand_v2
export FLAGS_selected_xpus=0

nohup python -u -m paddle.distributed.launch --log_level=DEBUG tools/train.py
      -c configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml
      --eval --amp -o use_gpu=false
      use_xpu=true
      pretrain_weights=./pretrained_model/ppyoloe_plus_crn_l_80e_coco.pdparams
      >> log-yoloe_l_20240408.log 2>&1 &