variable "region" {
  description = "Cloud sandboxes usually allow only us-east-1 or us-west-2."
  type        = string
  default     = "us-east-1"
}

variable "alert_email" {
  description = "Optional: email for SNS work-order alerts (confirm the subscription email before running the simulator)."
  type        = string
  default     = ""
}

variable "enable_sagemaker" {
  description = "Create a SageMaker execution role so `make train` can run training jobs in this account."
  type        = bool
  default     = false
}

variable "enable_data_lake" {
  description = "Firehose -> S3 -> Glue/Athena path. Set false if the sandbox blocks Firehose or Glue; live scoring still works."
  type        = bool
  default     = true
}
