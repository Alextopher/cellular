const uint32_t prng_step = 0xaf82e6c7;

uint16_t squares_rand(uint idx) {
  uint32_t state = constants.prng_seed * (uint32_t(idx * globalInvocationCount()) + uint32_t(globalInvocationIdx()));
  uint32_t c1 = state;
  uint32_t c2 = state + constants.prng_seed;
  state = state * state + c1;
  state = (state << 16) | (state >> 16);
  state = state * state + c2;
  state = (state << 16) | (state >> 16);
  state = state * state + c1;
  state = (state << 16) | (state >> 16);
  return uint16_t((state * state + c2) >> 16);
}

