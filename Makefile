# Server options, not necessary if working locally
SERVER_IP=
SERVER_PATH=
USER=

DEVICE=cuda

# Simu options
SRC_DATASET=mini_imagenet
TGT_DATASET=$(SRC_DATASET)


# Modules
CLS_TRANSFORMS=Trivial # Feature transformations used before feeding to the classifier
DET_TRANSFORMS=Trivial # Feature transformations used before feeding to the OOD detector

FEATURE_DETECTOR=none # OOD detector working on arbitrary features
PROBA_DETECTOR=none # OOD detector working on probabilistic output
CLASSIFIER=SimpleShot # Classification method for few-shot

# Model settings for main experiments
#   To run other experiments from the appendix `make benchmark_backbone_dataset'. 
BACKBONE=resnet12#
MODEL_SRC=feat# Origin of the model. For all timm models, use MODEL_SRC=url
TRAINING=standard# To differentiate between episodic and standard models


# DATA
DATADIR=data 
SPLIT=test
ID_QUERY=15
OOD_QUERY=15
BROAD=False
N_TASKS=1000
SHOTS=5# will iterate over these values
SEEDS=0 1 2#


# Misc
EXP=default # name of the folder in which results will be stored.
DEBUG=False # runs with small number of tasks 
SIMU_PARAMS=  # just in case you need to track some particular args in out.csv
OVERRIDE=False # used to override existing entries in out.csv
TUNE=""
ABLATE=""
VISU=False
SAVE_PREDICTIONS=False # only used to save model's predictions as a numpy file
DETECTOR_CONFIG_FILE="configs/detectors.yaml"#

# === Base recipes ===

extract:
	    for split in train val test; do \
			python -m src.compute_features \
				--backbone $(BACKBONE) \
				--src_dataset $(SRC_DATASET) \
				--tgt_dataset $(TGT_DATASET) \
				--data_dir $(DATADIR) \
		        --model_source $(MODEL_SRC) \
		        --training $(TRAINING) \
		        --override $(OVERRIDE) \
		        --debug $(DEBUG) \
				--split $${split} ;\
	    done ;\


run:
	for seed in $(SEEDS); do \
		for backbone in $(BACKBONE); do \
			for shot in $(SHOTS); do \
				python3 -m src.inference \
					--exp_name "$(EXP)/$(SRC_DATASET)/$(TGT_DATASET)/$(SPLIT)/$${backbone}/$${id_query}id_$${ood_query}ood/$(MODEL_SRC)/det=${DET_TRANSFORMS} cls=${CLS_TRANSFORMS}/$(VERSIONAME) $(CLASSIFIER) $(FEATURE_DETECTOR) $(PROBA_DETECTOR)/$${seed}" \
					--data_dir $(DATADIR) \
					--classifier $(CLASSIFIER) \
					--random_seed $${seed} \
					--n_tasks $(N_TASKS) \
					--n_shot $${shot} \
					--feature_detector $(FEATURE_DETECTOR) \
					--proba_detector $(PROBA_DETECTOR) \
					--detector_transforms  $(DET_TRANSFORMS) \
					--classifier_transforms  $(CLS_TRANSFORMS) \
					--visu_episode $(VISU) \
					--backbone $${backbone} \
					--model_source $(MODEL_SRC) \
					--training $(TRAINING) \
					--split $(SPLIT) \
					--src_dataset $(SRC_DATASET) \
					--n_id_query $(ID_QUERY) \
					--n_ood_query $(OOD_QUERY) \
					--broad_open_set $(BROAD) \
					--tgt_dataset $(TGT_DATASET) \
					--simu_hparams $(SIMU_PARAMS) \
					--override $(OVERRIDE) \
					--tune $(TUNE) \
					--ablate $(ABLATE) \
					--debug $(DEBUG) \
					--save_predictions $(SAVE_PREDICTIONS) \
					--device $(DEVICE) \
					--detector_config_file $(DETECTOR_CONFIG_FILE) ;\
			done ;\
		done ;\
	done ;\

# ========== Extraction pipelines ===========

extract_all:
	# Extract for RN and WRN
	for backbone in resnet12 wrn2810; do \
		for dataset in mini_imagenet tiered_imagenet; do \
			make BACKBONE=$${backbone} SRC_DATASET=$${dataset} MODEL_SRC='feat' TGT_DATASET=$${dataset} extract ;\
			make BACKBONE=$${backbone} TRAINING='feat' SRC_DATASET=$${dataset} MODEL_SRC='feat' TGT_DATASET=$${dataset} extract ;\
		done ;\
	done ;\

	# Tiered-Imagenet -> *
	for backbone in resnet12 wrn2810; do \
		for dataset in aircraft cub fungi; do \
			make BACKBONE=$${backbone} TRAINING='feat' SRC_DATASET=tiered_imagenet MODEL_SRC='feat' TGT_DATASET=$${dataset} extract ;\
			make BACKBONE=$${backbone} SRC_DATASET=tiered_imagenet MODEL_SRC='feat' TGT_DATASET=$${dataset} extract ;\
		done ;\
	done ;\

	# Imagenet -> *
	for dataset in fungi imagenet; do \
		for backbone in clip_vit_base_patch16 vit_base_patch16_224 vit_base_patch16_224_dino vit_base_patch16_224_sam resnet50 dino_resnet50 ssl_resnet50 swsl_resnet50 mixer_b16_224_in21k mixer_b16_224_miil_in21k; do \
			make BACKBONE=$${backbone} SRC_DATASET=imagenet MODEL_SRC='timm' TGT_DATASET=$${dataset} extract ;\
		done ;\
	done ;\


extract_bis:
	for backbone in resnet12 wrn2810; do \
			for split in train val test; do \
					python -m src.compute_features \
							--backbone $${backbone} \
							--src_dataset mini_imagenet \
							--tgt_dataset mini_imagenet_bis \
							--data_dir $(DATADIR) \
							--model_source feat \
							--training $(TRAINING) \
							--split $${split} \
							--keep_all_train_features True ;\
			done \
	done ;\


# ========== Feature Investigation ==========

clustering_metrics:
	for dataset in mini_imagenet tiered_imagenet; do \
			for split in train test; do \
					python -m src.investigate_features \
							data/features/$${dataset}/$${dataset}_bis/$${split}/standard/resnet12_$${dataset}_feat_4_4.pickle ;\
					python -m src.investigate_features \
							data/features/$${dataset}/$${dataset}_bis/$${split}/standard/wrn2810_$${dataset}_feat_last.pickle ;\
			done ;\
	done ;\

	for dataset in aircraft imagenet_val; do \
			for feature in ssl_resnext101_32x16d_imagenet_url_4_3 vit_base_patch16_224_in21k_imagenet_url_last_cls deit_tiny_patch16_224_imagenet_url_last_cls; do \
					python -m src.investigate_features \
							data/features/imagenet/$${dataset}/test/standard/$${feature}.pickle ;\
			done ;\
	done ;\


# ========== Running pipelines ===========

run_pyod:
	for method in HBOS KNN PCA OCSVM IForest COPOD; do \
		make CLS_TRANSFORMS="Pool BaseCentering L2norm" DET_TRANSFORMS="Pool BaseCentering L2norm" FEATURE_DETECTOR=$${method} run ;\
	done ;\

run_best:
	make run_osem ;\
	make run_snatcher ;\
	make CLS_TRANSFORMS="Pool BaseCentering L2norm" DET_TRANSFORMS="Pool BaseCentering L2norm" CLASSIFIER=SimpleShot FEATURE_DETECTOR=KNN run ;\
	make DET_TRANSFORMS="Pool BaseCentering L2norm" FEATURE_DETECTOR=OpenMax run ;\
	make CLS_TRANSFORMS="Pool MeanCentering L2norm" CLASSIFIER=TIM_GD PROBA_DETECTOR=MaxProbDetector run ;\

run_finalists:
	make run_osem ;\
	make CLS_TRANSFORMS="Pool BaseCentering L2norm" DET_TRANSFORMS="Pool BaseCentering L2norm" CLASSIFIER=SimpleShot FEATURE_DETECTOR=KNN run ;\

run_classifiers:
	for classifier in ICI LaplacianShot TIM_GD BDCSPN; do \
		make PROBA_DETECTOR=MaxProbDetector CLS_TRANSFORMS="Pool MeanCentering L2norm" CLASSIFIER=$${classifier} run ;\
	done ;\
	for classifier in Finetune SimpleShot; do \
		make PROBA_DETECTOR=MaxProbDetector CLS_TRANSFORMS="Pool BaseCentering L2norm" CLASSIFIER=$${classifier} run ;\
	done ;\
	make PROBA_DETECTOR=MaxProbDetector MODEL_SRC=feat TRAINING=feat CLASSIFIER=FEAT run ;\
	make CLS_TRANSFORMS="Pool Power QRreduction L2norm MeanCentering"  PROBA_DETECTOR=MaxProbDetector CLASSIFIER=MAP run ;\

run_snatcher:
	make MODEL_SRC=feat TRAINING=feat FEATURE_DETECTOR=SnatcherF run ;\

run_ostim:
	make FEATURE_DETECTOR=OSTIM run ;\

run_osem:
	make FEATURE_DETECTOR=OSEM DET_TRANSFORMS="Pool MeanCentering L2norm" run ;\

run_eol:
	make FEATURE_DETECTOR=EOL DET_TRANSFORMS="Pool BaseCentering L2norm" run ;\

run_open_set:
	for method in RPL PROSER OpenMax; do \
		make DET_TRANSFORMS="Pool BaseCentering L2norm" FEATURE_DETECTOR=$${method} run ;\
	done \

run_osfsl_t:
	make FEATURE_DETECTOR=OSTIM run ;\
	make FEATURE_DETECTOR=OSEM DET_TRANSFORMS="Pool MeanCentering L2norm" run ;\
	make FEATURE_DETECTOR=EOL DET_TRANSFORMS="Pool BaseCentering L2norm" run ;\


# ========== 0) Separation histogram ==========

simu_maxprob_hist:
	for split in test; do \
		for classifier in TIM_GD SimpleShot; do \
			make SHOTS=5 EXP=maxprob_hist SAVE_PREDICTIONS=True PROBA_DETECTOR=MaxProbDetector \
				CLS_TRANSFORMS="Pool BaseCentering L2norm" SPLIT=$${split} \
				SRC_DATASET=mini_imagenet TGT_DATASET=mini_imagenet_bis CLASSIFIER=$${classifier} run ;\
		done ;\
		make SHOTS=5 EXP=maxprob_hist SRC_DATASET=mini_imagenet TGT_DATASET=mini_imagenet_bis SPLIT=$${split} SAVE_PREDICTIONS=True run_ostim;\
	done ;\


maxprob_hist:
	for shot in 5; do \
		for split in test; do \
			python -m src.plots.torch_plotter \
				 --exp maxprob_hist \
				 --use_pretty False \
				 --filters n_shot=$${shot} split=$${split};\
		done ;\
	done ;\




# ========== 1) Validation ===========

tuning:
	make EXP=tuning TUNE=feature_detector SPLIT=val N_TASKS=500 run_osem ;\
	make EXP=tuning TUNE=classifier SPLIT=val N_TASKS=500 run_classifiers ;\
	make EXP=tuning TUNE=feature_detector SPLIT=val N_TASKS=500 run_open_set ;\
	make EXP=tuning TUNE=feature_detector SPLIT=val N_TASKS=500 run_pyod ;\
	make EXP=tuning TUNE=feature_detector SPLIT=val N_TASKS=500 run_snatcher ;\

log_best_configs:
	for shot in 1 5; do \
		python -m src.plots.csv_plotter \
			 --exp tuning \
			 --groupby classifier feature_detector \
			 --metrics mean_acc mean_rocauc \
			 --use_pretty False \
			 --plot_versus backbone \
			 --action log_best \
			 --filters n_shot=$${shot} ;\
	done ;\

# ========== 2) Standard benchmarks testing ===========

_benchmark:
	for dataset in mini_imagenet tiered_imagenet; do \
		make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} run_osem ;\
 		make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} run_snatcher ;\
 		make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} run_classifiers ;\
 		make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} run_pyod ;\
 		make SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} run_open_set ;\
	done ;\

benchmark:
	make EXP=benchmark _benchmark ;\

benchmark_broad_open_set:
	make EXP=benchmark_broad BROAD=True _benchmark ;\

log_benchmark:
	for dataset in mini_imagenet tiered_imagenet; do \
		for shot in 1 5 ; do \
			python -m src.plots.csv_plotter \
				 --exp benchmark \
				 --groupby classifier feature_detector \
				 --metrics mean_acc mean_rocauc mean_aupr mean_prec_at_90 \
				 --use_pretty True \
				 --plot_versus backbone \
				 --action log_latex \
				 --filters n_shot=$${shot} src_dataset=$${dataset} ;\
		done \
	done \

benchmark_backbone_dataset:
	make SRC_DATASET=mini_imagenet BACKBONE="resnet12 AGWR18ibn AGWR50ibn clip_vit_base_patch16 vit_base_patch16_224 efficientnet_b0" run_osfsl_t;\
	make SRC_DATASET=aircraft BACKBONE="clip_vit_base_patch16 vit_base_patch16_224 efficientnet_b0" run_osfsl_t;\
	make SRC_DATASET=cub BACKBONE="clip_vit_base_patch16 vit_base_patch16_224 efficientnet_b0" run_osfsl_t;\


# ========== 3) Cross-domain experiments ===========

spider_charts:
	# Tiered -> CUB
	for backbone in resnet12 wrn2810; do \
		make EXP=spider BACKBONE=$${backbone} run_best ;\
		for dataset in tiered_imagenet fungi aircraft cub; do \
			make EXP=spider BACKBONE=$${backbone} SRC_DATASET=tiered_imagenet TGT_DATASET=$${dataset} run_best ;\
		done ; \
	done ;\

plot_spider_charts:
	for shot in 1 5; do \
		for backbone in resnet12 wrn2810; do \
			python -m src.plots.spider_plotter \
				 --exp spider \
				 --groupby classifier feature_detector \
				 --use_pretty True \
				 --horizontal False \
				 --metrics mean_acc mean_rocauc mean_aupr mean_prec_at_90 \
				 --plot_versus src_dataset tgt_dataset \
				 --filters n_shot=$${shot} \
				 backbone=$${backbone} ;\
		done ;\
	done ;\

plot_main_spider_chart:
	python -m src.plots.spider_plotter \
		--exp spider \
		--groupby classifier feature_detector \
		--use_pretty True \
		--horizontal True \
		--metrics mean_acc mean_rocauc mean_aupr \
		--plot_versus src_dataset tgt_dataset \
		--filters n_shot=1 \
		backbone=resnet12 ;\


# ========== 4) Model agnosticity ==========


model_agnosticity:
	# Imagenet -> *
	for backbone in vit_base_patch16_224 clip_vit_base_patch16 vit_base_patch16_224_dino vit_base_patch16_224_sam resnet50 dino_resnet50 ssl_resnet50 swsl_resnet50 mixer_b16_224_in21k mixer_b16_224_miil_in21k; do \
		for dataset in fungi; do \
			make EXP=barplots SHOTS=1 MODEL_SRC='timm' BACKBONE=$${backbone} SRC_DATASET=imagenet TGT_DATASET=$${dataset} run_finalists ;\
		done ; \
	done ;\


plot_model_agnosticity:
	python -m src.plots.bar_plotter \
		 --exp barplots \
		 --groupby classifier feature_detector \
		 --metrics mean_acc mean_rocauc \
		 --latex True \
		 --plot_versus backbone \
		 --filters n_shot=1 ;\


# ========== 5) Ablation study ==========

ablate_osem:
	for dataset in mini_imagenet tiered_imagenet ; do \
		make EXP=ablation ABLATE=feature_detector SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} run_osem ;\
		make EXP=ablation/trivial SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} FEATURE_DETECTOR=OSEM DET_TRANSFORMS="Pool Trivial L2norm" run ;\
		make EXP=ablation/base SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} FEATURE_DETECTOR=OSEM DET_TRANSFORMS="Pool BaseCentering L2norm" run ;\
		make EXP=ablation/mean SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} FEATURE_DETECTOR=OSEM DET_TRANSFORMS="Pool MeanCentering L2norm" run ;\
	done ;\

ablate_ostim:
	# Imagenet -> *
	for dataset in mini_imagenet tiered_imagenet; do \
		make EXP=ablation ABLATE=feature_detector SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} run_ostim ;\
	done \

ablate_rebuttal:
	make EXP=ablation/rebuttal ABLATE=feature_detector SRC_DATASET=mini_imagenet TGT_DATASET=mini_imagenet run_osem ;\
	for dataset in tiered_imagenet fungi aircraft cub; do \
		make EXP=ablation/rebuttal ABLATE=feature_detector SRC_DATASET=tiered_imagenet TGT_DATASET=$${dataset} run_osem ;\
	done ; \

plot_ablation_rebuttal:
	for shot in 1 5; do \
		python -m src.plots.spider_plotter \
			 --exp ablation/rebuttal \
			 --groupby feature_detector \
			 --use_pretty True \
			 --ablation True \
			 --horizontal True \
			 --metrics mean_prototypes_similarity mean_acc mean_rocauc \
			 --plot_versus src_dataset tgt_dataset \
			 --filters n_shot=$${shot} \
			 backbone=resnet12 ;\
	done ;\

# ========== 6) Size of query set ==========

_variate_query:
	make run_finalists ;\
	for classifier in LaplacianShot TIM_GD BDCSPN; do \
		make PROBA_DETECTOR=MaxProbDetector CLS_TRANSFORMS="Pool MeanCentering L2norm" CLASSIFIER=$${classifier} run ;\
	done ;\
	make CLS_TRANSFORMS="Pool Power QRreduction L2norm MeanCentering"  PROBA_DETECTOR=MaxProbDetector CLASSIFIER=MAP run ;\

variate_query:
	for dataset in mini_imagenet tiered_imagenet; do \
		for query in 1 5 15 30; do \
			make EXP=variate_query/$${query} SRC_DATASET=$${dataset} TGT_DATASET=$${dataset} ID_QUERY=$${query} OOD_QUERY=$${query} _variate_query ;\
		done ;\
	done \

plot_variate_query:
	python -m src.plots.queries_plotter variate_query ; \

broad_open_set:
	make EXP=broad_open_set/true SRC_DATASET=mini_imagenet TGT_DATASET=mini_imagenet BROAD=True _variate_query ;\
	make EXP=broad_open_set/false SRC_DATASET=mini_imagenet TGT_DATASET=mini_imagenet BROAD=False _variate_query ;\

plot_broad_open_set:
	python -m src.plots.broad_plotter broad_open_set ; \

# ================= Deployment / Imports ==================

deploy:
	rsync -avm Makefile $(SERVER_IP):${SERVER_PATH}/ ;\
	rsync -avm --exclude '*.pyc' src $(SERVER_IP):${SERVER_PATH}/ ;\
	rsync -avm --exclude '*.pyc' scripts $(SERVER_IP):${SERVER_PATH}/ ;\
	rsync -avm --exclude '*.pyc' configs $(SERVER_IP):${SERVER_PATH}/ ;\

import/results:
	rsync -avm $(SERVER_IP):${SERVER_PATH}/results ./ ;\

import/features:
	rsync -avm $(SERVER_IP):${SERVER_PATH}/data/features ./data/ ;\

import/archive:
	rsync -avm $(SERVER_IP):${SERVER_PATH}/archive ./ ;\

import/tiered:
	rsync -avm $(SERVER_IP):${SERVER_PATH}/data/tiered_imagenet.tar.gz ./data/ ;\

import/models:
	for dataset in mini_imagenet tiered_imagenet fgvc-aircraft-2013b cub; do \
		rsync -avm $(SERVER_IP):${SERVER_PATH}/data/models .;\
	done ;\

tar_data:
	for dataset in mini_imagenet tiered_imagenet fgvc-aircraft-2013b cub; do \
		tar -czvf  data/$${dataset}.tar.gz -C data/ $${dataset} ;\
	done ;\


deploy_data:
	for dataset in mini_imagenet tiered_imagenet fgvc-aircraft-2013b cub; do \
		rsync -avm data/$${dataset}.tar.gz $(SERVER_IP):${SERVER_PATH}/data/ ;\
	done ;\

deploy_models:
	for dataset in mini_imagenet tiered_imagenet fgvc-aircraft-2013b cub; do \
		rsync -avm data/models $(SERVER_IP):${SERVER_PATH}/ ;\
	done ;\


deploy_features:
	for dataset in mini_imagenet tiered_imagenet fgvc-aircraft-2013b cub; do \
		rsync -avm data/features/$${dataset} $(SERVER_IP):${SERVER_PATH}/data/features/ ;\
	done ;\


kill_all: ## Kill all my python and tee processes on the server
	ps -u $(USER) | grep "python" | sed 's/^ *//g' | cut -d " " -f 1 | xargs kill
	ps -u $(USER) | grep "tee" | sed 's/^ *//g' | cut -d " " -f 1 | xargs kill


# ============= Downlooad/Prepare data ============

aircraft:
	mkdir -p data
	wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
	tar -xvf  fgvc-aircraft-2013b.tar.gz -C data ;\
	rm fgvc-aircraft-2013b.tar.gz ;\

fungi:
	mkdir -p data/fungi ;\
	wget https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz ;\
	wget https://labs.gbif.org/fgvcx/2018/train_val_annotations.tgz ;\
	tar -xvf fungi_train_val.tgz -C data/fungi ;\
	tar -xvf train_val_annotations.tgz -C data/fungi ;\
	rm fungi_train_val.tgz; rm train_val_annotations.tgz ;

cub:
	mkdir -p data/cub ;\
	wget https://data.caltech.edu/tindfiles/serve/1239ea37-e132-42ee-8c09-c383bb54e7ff/


mini_imagenet_bis:
	python -m scripts.generate_mini_imagenet_bis


# ============= Archive results =============

archive: # Archive experiments
	python src/utils/list_files.py results/ archive/ tmp.txt
	{ read -r out_files; read -r archive_dir; } < tmp.txt ; \
	for file in $${out_files}; do \
		cp -Rv $${file} $${archive_dir}/ ; \
	done
	rm tmp.txt


restore: # Restore experiments to output/
	python src/utils/list_files.py archive/ results/ tmp.txt ; \
	read -r out_files < tmp.txt ; \
	folder=`echo ${out_files} | cut -d'/' -f2-` ;\
	mkdir -p results/$${folder} ; \
	for file in $${out_files}; do \
		cp -Rv $${file} results/$${folder}/ ; \
	done
	rm tmp.txt
