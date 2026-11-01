terraform {
  required_version = ">= 1.5"
  required_providers {
    aws     = { source = "hashicorp/aws", version = ">= 5.60" }
    archive = { source = "hashicorp/archive", version = ">= 2.4" }
    local   = { source = "hashicorp/local", version = ">= 2.4" }
    random  = { source = "hashicorp/random", version = ">= 3.6" }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "ev-pdm-poc" }
  }
}

resource "random_id" "suffix" {
  byte_length = 3
}

locals {
  name = "ev-pdm-${random_id.suffix.hex}"
}

data "aws_caller_identity" "me" {}

# ---------- Storage: data lake bucket (also holds the learned baseline) ----------

resource "aws_s3_bucket" "lake" {
  bucket        = "${local.name}-lake-${data.aws_caller_identity.me.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "lake" {
  bucket                  = aws_s3_bucket.lake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------- State: component health + work orders ----------

resource "aws_dynamodb_table" "health" {
  name         = "${local.name}-health"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "vehicle_id"
  range_key    = "component"

  attribute {
    name = "vehicle_id"
    type = "S"
  }
  attribute {
    name = "component"
    type = "S"
  }
}

resource "aws_dynamodb_table" "work_orders" {
  name         = "${local.name}-work-orders"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "work_order_id"

  attribute {
    name = "work_order_id"
    type = "S"
  }
}

# ---------- Alerts ----------

resource "aws_sns_topic" "alerts" {
  name = "${local.name}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}
