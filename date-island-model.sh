python date-island-model.py \
  --relate-dir ../relate \
  --relate-lib-dir ../relate_lib \
  --ep-iterations 25 \
  --overwrite-from-ep \
  --output-dir ./island-model &>>island-model.log

#python date-island-model.py \
#  --relate-dir ../relate \
#  --relate-lib-dir ../relate_lib \
#  --mutation-rate 6.5e-9 \
#  --output-dir ./island-model-lowmut &>>island-model-lowmut.log
#
#python date-island-model.py \
#  --relate-dir ../relate \
#  --relate-lib-dir ../relate_lib \
#  --num-contemporary 100 400 \
#  --output-dir ./island-model-unbal &>>island-model-unbal.log
#
#python date-island-model.py \
#  --relate-dir ../relate \
#  --relate-lib-dir ../relate_lib \
#  --num-contemporary 125 125 \
#  --num-ancient 125 125 \
#  --ancients-ages-unknown \
#  --output-dir ./island-model-dateanc &>>island-model-dateanc.log
#
#python date-island-model.py \
#  --relate-dir ../relate \
#  --relate-lib-dir ../relate_lib \
#  --num-contemporary 125 125 \
#  --num-ancient 125 125 \
#  --output-dir ./island-model-fixanc &>island-model-fixanc.log
