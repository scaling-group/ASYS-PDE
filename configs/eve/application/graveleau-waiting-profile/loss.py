"""Editable initial fitting loss; protected scoring uses different fixed samples."""
def loss_fn(model, batch, task):
    terms = task.terms(model, batch)
    return sum(terms.values())
