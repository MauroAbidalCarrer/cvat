import json
import base64


def init_context(context):
    context.logger.info("SAM2 Video Tracker initialized (logging mode)")
    print(context.__dict__.keys())

def handler(context, event):
    data = event.body
    if isinstance(data, (bytes, bytearray)):
        data = json.loads(data)

    shapes = data.get("shapes", [])
    states = data.get("states", [])

    context.logger.info(f"=== TRACKER REQUEST ===")
    print(context.__dict__.keys())
    print(event.__dict__.keys())
    context.logger.info(context)
    context.logger.info(event)
    context.logger.info(f"Shapes: {shapes}")
    context.logger.info(f"States: {states}")
    context.logger.info(f"Has image: {'image' in data}")

    if not states:
        # First call: initialization
        context.logger.info(f"INIT: box={shapes[0] if shapes else 'none'}")
        return context.Response(
            body=json.dumps({
                "shapes": shapes,
                "states": [{"initialized": True}] * len(shapes),
            }),
            headers={},
            content_type="application/json",
            status_code=200,
        )
    else:
        # Subsequent calls: tracking
        context.logger.info(f"TRACK: states={states}")
        return context.Response(
            body=json.dumps({
                "shapes": shapes,
                "states": states,
            }),
            headers={},
            content_type="application/json",
            status_code=200,
        )

