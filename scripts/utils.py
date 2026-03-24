import logging
import os
import sys


def setup_sumo_home():
    if 'SUMO_HOME' not in os.environ:
        try:
            import sumo  # use eclipse-sumo package
            os.environ['SUMO_HOME'] = sumo.SUMO_HOME
        except ImportError:
            logging.error("SUMO_HOME environment variable not set and eclipse-sumo is not installed. \n"
                          "Please either install eclipse-sumo or set SUMO_HOME variable to your SUMO installation directory.")
            sys.exit("Error: SUMO_HOME environment variable not set and eclipse-sumo is not installed. \n"
                     "Please either install eclipse-sumo or set SUMO_HOME variable to your SUMO installation directory.")

    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    logging.info("SUMO_HOME found and tools added to path.")
