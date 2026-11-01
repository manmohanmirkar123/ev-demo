# ---------- Scorer Lambda: anomaly + RUL per component, writes health, emits risk events ----------

data "archive_file" "scorer" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda/scorer"
  output_path = "${path.module}/.build/scorer.zip"
  excludes    = ["__pycache__"]
}

resource "aws_iam_role" "scorer" {
  name = "${local.name}-scorer"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "scorer_logs" {
  role       = aws_iam_role.scorer.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "scorer" {
  role = aws_iam_role.scorer.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["dynamodb:BatchGetItem", "dynamodb:BatchWriteItem", "dynamodb:PutItem"]
        Resource = aws_dynamodb_table.health.arn
      },
      {
        Effect   = "Allow"
        Action   = "events:PutEvents"
        Resource = "arn:aws:events:${var.region}:${data.aws_caller_identity.me.account_id}:event-bus/default"
      },
      {
        Effect   = "Allow"
        Action   = "s3:GetObject"
        Resource = ["${aws_s3_bucket.lake.arn}/baseline/*", "${aws_s3_bucket.lake.arn}/model/*"]
      },
      {
        # Lets the handler tell "no baseline yet" (NoSuchKey) apart from AccessDenied.
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = aws_s3_bucket.lake.arn
      },
    ]
  })
}

resource "aws_lambda_function" "scorer" {
  function_name    = "${local.name}-scorer"
  role             = aws_iam_role.scorer.arn
  runtime          = "python3.12"
  handler          = "handler.handler"
  filename         = data.archive_file.scorer.output_path
  source_code_hash = data.archive_file.scorer.output_base64sha256
  timeout          = 15
  memory_size      = 256

  environment {
    variables = {
      HEALTH_TABLE    = aws_dynamodb_table.health.name
      BASELINE_BUCKET = aws_s3_bucket.lake.bucket
    }
  }
}

resource "aws_lambda_permission" "iot" {
  statement_id  = "AllowIoTRule"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scorer.function_name
  principal     = "iot.amazonaws.com"
  source_arn    = aws_iot_topic_rule.telemetry.arn
}
