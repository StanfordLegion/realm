# Realm Python Bindings

## Installation instructions

```bash
pip install -v .
```

## Example

```python
import realm

r = realm.Runtime.get_runtime()
r.init()

e = realm.UserEvent.create_user_event()
r.shutdown(e)
e.trigger()
r.wait_for_shutdown()
```