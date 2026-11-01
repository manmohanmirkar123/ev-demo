# ---------- Device identity for the simulator (stands in for the vehicle TCU) ----------

data "aws_iot_endpoint" "ats" {
  endpoint_type = "iot:Data-ATS"
}

resource "aws_iot_thing" "simulator" {
  name = "${local.name}-fleet-simulator"
}

resource "aws_iot_certificate" "simulator" {
  active = true
}

resource "aws_iot_policy" "simulator" {
  name = "${local.name}-simulator"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "iot:Connect"
        Resource = "arn:aws:iot:${var.region}:${data.aws_caller_identity.me.account_id}:client/ev-sim-*"
      },
      {
        Effect   = "Allow"
        Action   = "iot:Publish"
        Resource = "arn:aws:iot:${var.region}:${data.aws_caller_identity.me.account_id}:topic/ev/telemetry/*"
      },
    ]
  })
}

resource "aws_iot_policy_attachment" "simulator" {
  policy = aws_iot_policy.simulator.name
  target = aws_iot_certificate.simulator.arn
}

resource "aws_iot_thing_principal_attachment" "simulator" {
  thing     = aws_iot_thing.simulator.name
  principal = aws_iot_certificate.simulator.arn
}

resource "local_file" "cert" {
  filename        = "${path.module}/../simulator/certs/device.pem.crt"
  content         = aws_iot_certificate.simulator.certificate_pem
  file_permission = "0600"
}

resource "local_sensitive_file" "key" {
  filename        = "${path.module}/../simulator/certs/private.pem.key"
  content         = aws_iot_certificate.simulator.private_key
  file_permission = "0600"
}

# ---------- Routing: every telemetry message -> scorer Lambda (+ data lake) ----------

resource "aws_iot_topic_rule" "telemetry" {
  name        = replace("${local.name}_telemetry", "-", "_")
  enabled     = true
  sql         = "SELECT * FROM 'ev/telemetry/+'"
  sql_version = "2016-03-23"

  lambda {
    function_arn = aws_lambda_function.scorer.arn
  }

  dynamic "firehose" {
    for_each = var.enable_data_lake ? [1] : []
    content {
      delivery_stream_name = aws_kinesis_firehose_delivery_stream.telemetry[0].name
      role_arn             = aws_iam_role.iot_rule[0].arn
      separator            = "\n"
    }
  }
}

resource "aws_iam_role" "iot_rule" {
  count = var.enable_data_lake ? 1 : 0
  name  = "${local.name}-iot-rule"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "iot.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "iot_rule" {
  count = var.enable_data_lake ? 1 : 0
  role  = aws_iam_role.iot_rule[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["firehose:PutRecord", "firehose:PutRecordBatch"]
      Resource = aws_kinesis_firehose_delivery_stream.telemetry[0].arn
    }]
  })
}
