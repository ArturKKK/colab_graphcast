# @title train model
!pip install optax
import jax
import jax.numpy as jnp
import optax
learning_rate = 1e-4
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)
num_steps = 100  # Общее число шагов обновления (итераций)

for step in range(num_steps):
    # 1) Вычисляем лосс и градиенты на текущем батче
    loss, diagnostics, next_state, grads = grads_fn_jitted(
        params=params,
        state=state,
        inputs=train_inputs,
        targets=train_targets,
        forcings=train_forcings
    )
    
    # 2) Обновляем параметры (weights) с помощью Adam
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # 3) Обновляем state (если модель хранит внутреннее состояние, напр. BatchNorm)
    state = next_state
    
    # 4) Печатаем лосс каждые 10 шагов, чтобы видеть динамику
    if step % 10 == 0:
        print(f"Step {step}, Loss = {loss:.4f}") 
