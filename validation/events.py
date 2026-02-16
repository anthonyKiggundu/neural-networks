import torch

def action_to_event_vector(proposal_json, context):
    """
    Converts telecom action proposal -> 16D event vector.
    """
    e = torch.zeros(16)

    if proposal_json["action"] == "offload_traffic":
        e[0] = proposal_json["percentage"]

    e[1] = context["hour"] / 24.0
    e[2] = context["day_of_week"] / 7.0

    return e.unsqueeze(0)

