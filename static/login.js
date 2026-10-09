
      const form = document.querySelector('#login-form');
      const error = document.querySelector('#login-error');
      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        error.textContent = '';
        const button = form.querySelector('button');
        button.disabled = true;
        try {
          const response = await fetch('/api/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            signal: AbortSignal.timeout(15000),
            body: JSON.stringify({password: document.querySelector('#password').value}),
          });
          const body = await response.json();
          if (!response.ok) throw new Error(body.error || 'Falha na autenticação.');
          window.location.assign('/');
        } catch (err) {
          error.textContent = err.message;
          button.disabled = false;
        }
      });
