#!/bin/bash

SKIPCONDA=0
SHOWHELP=0

usage() {
				echo "Usage: . ./setup.sh [--skip-conda | --help]"
				exit 1
}

opts=$(getopt -o sh --long skip-conda,help -- "$@")

eval set -- "$opts"
while :
do
	case "$1" in 
		--skip-conda) SKIPCONDA=1 ; shift ;;
		--help) SHOWHELP=1 ; shift ;;

		--) shift; break ;;
	esac
done

if [ "$SHOWHELP" -eq 1 ]; then
	usage
fi

if [ "$SKIPCONDA" -eq 0 ]; then
	echo "Installing Anaconda"
	sudo apt install -y \
					libgl1-mesa-glx \
					libegl1-mesa \
					libxrandr2 \
					libxrandr2 \
					libxss1 \
					libxcursor1 \
					libxcomposite1 \
					libasound2 \
					libxi6 \
					libxtst6
	
	curl https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh > ./bin/Anaconda3-2021.05-Linux-x86_64.sh
	bash ./bin/Anaconda3-2021.05-Linux-x86_64.sh

	alias conda="./bin/anaconda3/condabin/conda"

else
	echo "Skipping conda installation"
fi

export PYTHONPATH=~/Documents/mlflow/
