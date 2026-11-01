# ---------- Act: EventBridge -> Step Functions -> work order (DynamoDB) + notification (SNS) ----------

resource "aws_iam_role" "sfn" {
  name = "${local.name}-sfn"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "states.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "sfn" {
  role = aws_iam_role.sfn.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = "dynamodb:PutItem", Resource = aws_dynamodb_table.work_orders.arn },
      { Effect = "Allow", Action = "sns:Publish", Resource = aws_sns_topic.alerts.arn },
    ]
  })
}

resource "aws_sfn_state_machine" "service" {
  name     = "${local.name}-book-service"
  role_arn = aws_iam_role.sfn.arn

  definition = jsonencode({
    Comment = "Turn a component-at-risk event into a prioritised service work order"
    StartAt = "Prioritise"
    States = {
      Prioritise = {
        Type    = "Choice"
        Choices = [{ Variable = "$.detail.status", StringEquals = "CRITICAL", Next = "Urgent" }]
        Default = "Planned"
      }
      Urgent = {
        Type       = "Pass"
        Parameters = { "priority" = "P1", "book_within" = "48 hours", "id.$" = "States.UUID()" }
        ResultPath = "$.plan"
        Next       = "CreateWorkOrder"
      }
      Planned = {
        Type       = "Pass"
        Parameters = { "priority" = "P2", "book_within" = "14 days", "id.$" = "States.UUID()" }
        ResultPath = "$.plan"
        Next       = "CreateWorkOrder"
      }
      CreateWorkOrder = {
        Type     = "Task"
        Resource = "arn:aws:states:::dynamodb:putItem"
        Parameters = {
          TableName = aws_dynamodb_table.work_orders.name
          Item = {
            work_order_id = { "S.$" = "$.plan.id" }
            vehicle_id    = { "S.$" = "$.detail.vehicle_id" }
            component     = { "S.$" = "$.detail.label" }
            status        = { "S.$" = "$.detail.status" }
            priority      = { "S.$" = "$.plan.priority" }
            book_within   = { "S.$" = "$.plan.book_within" }
            rul_days      = { "S.$" = "$.detail.rul_text" }
            reading       = { "S.$" = "States.Format('{} {}', $.detail.value, $.detail.unit)" }
            sim_day       = { "S.$" = "States.Format('{}', $.detail.sim_day)" }
            created_at    = { "S.$" = "$$.State.EnteredTime" }
            state         = { "S" = "OPEN" }
          }
        }
        ResultPath = null
        Next       = "Notify"
      }
      Notify = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn    = aws_sns_topic.alerts.arn
          "Subject.$" = "States.Format('[{}] {} - {}', $.plan.priority, $.detail.vehicle_id, $.detail.label)"
          "Message.$" = "States.Format('{} on {} is {} (reading {} {}, est. {} days to limit). Book service within {}. Work order {}.', $.detail.label, $.detail.vehicle_id, $.detail.status, $.detail.value, $.detail.unit, $.detail.rul_text, $.plan.book_within, $.plan.id)"
        }
        End = true
      }
    }
  })
}

resource "aws_cloudwatch_event_rule" "at_risk" {
  name = "${local.name}-component-at-risk"
  event_pattern = jsonencode({
    source        = ["ev.pdm"]
    "detail-type" = ["ComponentAtRisk"]
  })
}

resource "aws_iam_role" "events" {
  name = "${local.name}-events"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "events.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "events" {
  role = aws_iam_role.events.id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = "states:StartExecution", Resource = aws_sfn_state_machine.service.arn }]
  })
}

resource "aws_cloudwatch_event_target" "sfn" {
  rule     = aws_cloudwatch_event_rule.at_risk.name
  arn      = aws_sfn_state_machine.service.arn
  role_arn = aws_iam_role.events.arn
}
