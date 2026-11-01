PY      := .venv/bin/python
TF      := terraform -chdir=infra
CA_URL  := https://www.amazontrust.com/repository/AmazonRootCA1.pem

# Two AWS accounts: the fleet stack (IoT, Lambda, ...) and SageMaker training. Override if your
# profile names differ, e.g. make deploy FLEET_PROFILE=acg
FLEET_PROFILE ?= default
SM_PROFILE    ?= kodekloud
SM_REGION     ?= us-east-1
FLEET         := AWS_PROFILE=$(FLEET_PROFILE)
SM            := AWS_PROFILE=$(SM_PROFILE)

.PHONY: help venv test evaluate check deploy simulate baseline train train-local train-remote bundle publish-model unpublish-model \
        dashboard reset destroy

help:
	@echo "Modes (all in the fleet account unless noted):"
	@echo "  A  no ML               make deploy                -> linear trend"
	@echo "  B  ML, no SageMaker    make deploy; make train-local"
	@echo "  C  ML with SageMaker   make deploy SAGEMAKER=1; make train"
	@echo "  D  separate ML acct    make deploy; make train-remote (CLI keys) or make bundle (console only)"
	@echo ""
	@echo "Local (no AWS):"
	@echo "  make venv             create .venv and install Python deps"
	@echo "  make test             offline tests"
	@echo "  make evaluate         train XGBoost locally and compare it with the linear trend"
	@echo "Fleet account (profile $(FLEET_PROFILE)):"
	@echo "  make check            confirm the profile works"
	@echo "  make deploy           terraform apply [SAGEMAKER=1] [EMAIL=you@example.com]"
	@echo "  make simulate         stream 120 simulated days of fleet telemetry to IoT Core"
	@echo "  make baseline         learn the healthy-fleet baseline with Athena"
	@echo "  make train-local      mode B: train XGBoost on this machine and publish it"
	@echo "  make train            mode C: SageMaker training job in this account, published automatically"
	@echo "  make unpublish-model  back to the linear trend (mode A)"
	@echo "  make dashboard        live Streamlit dashboard"
	@echo "  make reset            clear health + work orders to replay the demo"
	@echo "  make destroy          tear the fleet stack down"
	@echo "Separate ML account (mode D, profile $(SM_PROFILE)):"
	@echo "  make train-remote     SageMaker job in that account -> ml/.model/ev-pdm-model.json"
	@echo "  make bundle           zip for a console-only account: train in CloudShell / Studio"
	@echo "  make publish-model    copy the model into the fleet account [FROM=path/to/ev-pdm-model.json]"

venv:
	python3 -m venv .venv
	.venv/bin/pip install -q -r requirements.txt

test:
	$(PY) -m pytest -q tests

evaluate:
	$(PY) ml/evaluate.py

check:
	@echo "fleet account ($(FLEET_PROFILE)):"; $(FLEET) aws sts get-caller-identity --query Account --output text

deploy:
	$(FLEET) $(TF) init -upgrade -input=false
	$(FLEET) $(TF) apply -auto-approve -input=false $(if $(EMAIL),-var alert_email=$(EMAIL)) \
		$(if $(SAGEMAKER),-var enable_sagemaker=true)
	$(FLEET) $(TF) output -json > config.json
	curl -fsSL $(CA_URL) -o simulator/certs/AmazonRootCA1.pem
	@echo "Deployed. Next: make dashboard (new terminal), then make simulate"

simulate:
	$(FLEET) $(PY) simulator/simulate.py $(ARGS)

baseline:
	$(FLEET) $(PY) ml/build_baseline.py $(ARGS)

train:
	$(FLEET) $(PY) ml/train_sagemaker.py --config config.json $(ARGS)

train-local:
	$(FLEET) $(PY) ml/train_local.py $(ARGS)

train-remote:
	$(SM) $(PY) ml/train_sagemaker.py --region $(SM_REGION) $(ARGS)

# Only what training needs (stdlib + boto3), keeping the repo layout so the script's imports resolve.
BUNDLE_FILES := ml/train_sagemaker.py ml/dataset.py ml/sm_code/xgb_train.py ml/sm_code/hyperparams.py \
                lambda/scorer/health.py lambda/scorer/tree_model.py simulator/fleet.py
bundle:
	mkdir -p dist
	rm -f dist/ev-pdm-sagemaker.zip
	zip -q dist/ev-pdm-sagemaker.zip $(BUNDLE_FILES)
	@echo "dist/ev-pdm-sagemaker.zip → upload in CloudShell (Actions → Upload file), then:"
	@echo "  mkdir -p ev-pdm && cd ev-pdm && unzip -o ~/ev-pdm-sagemaker.zip && python3 ml/train_sagemaker.py"

publish-model:
	$(FLEET) $(PY) ml/publish_model.py "$(FROM)"

unpublish-model:
	$(FLEET) $(PY) ml/unpublish_model.py

dashboard:
	$(FLEET) .venv/bin/streamlit run dashboard/app.py

reset:
	$(FLEET) $(PY) scripts/reset_state.py

destroy:
	$(FLEET) $(TF) destroy -auto-approve -input=false
	rm -f config.json simulator/certs/*.pem* simulator/certs/*.key
