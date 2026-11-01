output "region" {
  value = var.region
}

output "iot_endpoint" {
  value = data.aws_iot_endpoint.ats.endpoint_address
}

output "health_table" {
  value = aws_dynamodb_table.health.name
}

output "work_orders_table" {
  value = aws_dynamodb_table.work_orders.name
}

output "lake_bucket" {
  value = aws_s3_bucket.lake.bucket
}

output "glue_database" {
  value = var.enable_data_lake ? aws_glue_catalog_database.lake[0].name : ""
}

output "athena_workgroup" {
  value = var.enable_data_lake ? aws_athena_workgroup.lake[0].name : ""
}

output "state_machine" {
  value = aws_sfn_state_machine.service.name
}

output "sagemaker_role_arn" {
  value = var.enable_sagemaker ? aws_iam_role.sagemaker[0].arn : ""
}

output "scorer_function" {
  value = aws_lambda_function.scorer.function_name
}
