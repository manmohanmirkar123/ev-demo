# ---------- Optional: SageMaker training in this same account (make deploy SAGEMAKER=1) ----------
# Training jobs read data from and write artifacts to s3://<lake>/sagemaker/; ml/train_sagemaker.py --config
# then publishes the model straight to s3://<lake>/model/ for the scorer.

resource "aws_iam_role" "sagemaker" {
  count = var.enable_sagemaker ? 1 : 0
  name  = "${local.name}-sagemaker"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "sagemaker.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "sagemaker" {
  count = var.enable_sagemaker ? 1 : 0
  role  = aws_iam_role.sagemaker[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = "${aws_s3_bucket.lake.arn}/sagemaker/*"
      },
      {
        Effect    = "Allow"
        Action    = "s3:ListBucket"
        Resource  = aws_s3_bucket.lake.arn
        Condition = { StringLike = { "s3:prefix" = ["sagemaker/*"] } }
      },
      {
        # Logs, metrics and pulling the AWS-owned XGBoost image.
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams",
          "cloudwatch:PutMetricData", "ecr:GetAuthorizationToken", "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability",
        ]
        Resource = "*"
      },
    ]
  })
}
