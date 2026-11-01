# ---------- Firehose -> S3 (partitioned by day) -> Glue table -> Athena ----------

resource "aws_iam_role" "firehose" {
  count = var.enable_data_lake ? 1 : 0
  name  = "${local.name}-firehose"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "firehose.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "firehose" {
  count = var.enable_data_lake ? 1 : 0
  role  = aws_iam_role.firehose[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:PutObject", "s3:GetBucketLocation", "s3:ListBucket", "s3:AbortMultipartUpload"]
      Resource = [aws_s3_bucket.lake.arn, "${aws_s3_bucket.lake.arn}/*"]
    }]
  })
}

resource "aws_kinesis_firehose_delivery_stream" "telemetry" {
  count       = var.enable_data_lake ? 1 : 0
  name        = "${local.name}-telemetry"
  destination = "extended_s3"

  extended_s3_configuration {
    role_arn            = aws_iam_role.firehose[0].arn
    bucket_arn          = aws_s3_bucket.lake.arn
    prefix              = "telemetry/dt=!{timestamp:yyyy-MM-dd}/"
    error_output_prefix = "errors/!{firehose:error-output-type}/"
    buffering_size      = 1
    buffering_interval  = 60
  }
}

resource "aws_glue_catalog_database" "lake" {
  count = var.enable_data_lake ? 1 : 0
  name  = replace("${local.name}_lake", "-", "_")
}

resource "aws_glue_catalog_table" "telemetry" {
  count         = var.enable_data_lake ? 1 : 0
  name          = "telemetry"
  database_name = aws_glue_catalog_database.lake[0].name
  table_type    = "EXTERNAL_TABLE"

  # Partition projection: new days are queryable immediately, no crawler needed.
  parameters = {
    "classification"            = "json"
    "projection.enabled"        = "true"
    "projection.dt.type"        = "date"
    "projection.dt.format"      = "yyyy-MM-dd"
    "projection.dt.range"       = "2025-01-01,NOW"
    "storage.location.template" = "s3://${aws_s3_bucket.lake.bucket}/telemetry/dt=$${dt}/"
  }

  partition_keys {
    name = "dt"
    type = "string"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.lake.bucket}/telemetry/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

    ser_de_info {
      serialization_library = "org.openx.data.jsonserde.JsonSerDe"
    }

    columns {
      name = "vehicle_id"
      type = "string"
    }
    columns {
      name = "ts"
      type = "string"
    }
    columns {
      name = "sim_day"
      type = "int"
    }
    columns {
      name = "odometer_km"
      type = "double"
    }
    columns {
      name = "motor"
      type = "struct<temp_c:double,vibration_g:double,rpm:int>"
    }
    columns {
      name = "inverter"
      type = "struct<temp_c:double>"
    }
    columns {
      name = "cooling"
      type = "struct<coolant_temp_c:double,pump_flow_lpm:double>"
    }
    columns {
      name = "brakes"
      type = "struct<pad_mm_front:double,pad_mm_rear:double>"
    }
    columns {
      name = "tyres"
      type = "struct<fl:double,fr:double,rl:double,rr:double>"
    }
    columns {
      name = "dtc"
      type = "array<string>"
    }
  }
}

resource "aws_athena_workgroup" "lake" {
  count         = var.enable_data_lake ? 1 : 0
  name          = "${local.name}-wg"
  force_destroy = true

  configuration {
    result_configuration {
      output_location = "s3://${aws_s3_bucket.lake.bucket}/athena-results/"
    }
  }
}
