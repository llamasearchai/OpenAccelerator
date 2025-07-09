import asyncio
import logging

from src.open_accelerator.core.accelerator import HighEnergyAccelerator
from src.open_accelerator.core.beam import BeamParameters

logging.basicConfig(level=logging.INFO)


async def main():
    try:
        # Initialize accelerator
        accelerator = HighEnergyAccelerator(max_energy=100.0, beam_current=1.0)

        # Create beam parameters
        beam = BeamParameters(energy=50.0, current=0.5, emittance=1e-6)

        # Run simulation
        result = await accelerator.simulate(beam)
        print(f"Simulation completed: {result}")

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
