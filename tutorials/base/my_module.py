from absl import app as absl_app
from absl import flags

import flag_source

flags.adopt_module_key_flags(flag_source)

def main(_):
  pass

absl_app.run(main)