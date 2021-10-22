#!/bin/zsh
#SBATCH --job-name=sep-train-pcc
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=5
#SBATCH -G 1 -c 10
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in
#SBATCH -w gnode58

#module load cuda/11.0
#module load cudnn/7-cuda-11.0

DATASET_FILE1="/share1/vikrant.dewangan/dataset/pcc/pointda/"
DATASET_DST="/scratch/shapenets"

function setup {
    mkdir -p $DATASET_DST
    echo "Inside setup"

    scp -r ada:$DATASET_FILE1 $DATASET_DST
}

echo "Copying"
#[ -d $DATASET_DST  ] || setup
setup
cd 

echo "Done copying"
#conda activate forecasting
echo "Done conda"

cd /scratch/shapenets/pointda

unzip PointDA_data.zip

cd ~/pcc/DefRec_and_PCM/
wandb agent -p pcc_sweep -e pcc-team --count 1 rz21p67j

echo "Done Training"
