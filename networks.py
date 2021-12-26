import jax as jx
import jax.numpy as jnp
import haiku as hk

min_denom = 0.000001
activation_dict = {"silu":jx.nn.silu, "elu": jx.nn.elu}

class recurrent_network(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config['num_hidden_units']
        self.activation_function = activation_dict[config['activation']]

    def __call__(self, phi, a, h):
        x = jnp.concatenate([hk.Flatten()(phi), a], axis=1)
        #GRU returns a 2-tuple but I belive both elements are the same, just return the nsext hidden state
        return hk.GRU(self.num_hidden_units)(x,h)[1]

class reward_network(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_hidden_units = config['num_hidden_units']
        self.activation_function = activation_dict[config['activation']]
        self.learn_reward_variance = config['learn_reward_variance']

    def __call__(self, phi, h, key=None):
        x = jnp.concatenate([hk.Flatten()(phi),h],axis=1)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        mu = hk.Linear(1)(x)[:,0]
        if(self.learn_reward_variance):
            sigma = jx.nn.softplus(hk.Linear(1)(x))[:,0]
            sigma = jnp.clip(sigma, min_denom, None)
        else:
            sigma = jnp.ones(mu.shape)
        return {'mu':mu, 'sigma':sigma}

class termination_network(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_hidden_units = config['num_hidden_units']
        self.activation_function = activation_dict[config['activation']]

    def __call__(self, phi, h, key=None):
        x = jnp.concatenate([hk.Flatten()(phi),h],axis=1)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        logit = hk.Linear(1)(x)[:,0]
        return {'logit':logit}

class next_phi_network(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_features = config['num_features']
        self.num_hidden_units = config['num_hidden_units']
        self.feature_width = config['feature_width']
        self.activation_function = activation_dict[config['activation']]
        self.latent_type = config['latent_type']

    def __call__(self, h, key):
        x = h
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))

        if(self.latent_type=='gaussian'):
            mu = hk.Linear(self.num_features)(hk.Flatten()(x))
            sigma = jx.nn.softplus(hk.Linear(self.num_features)(hk.Flatten()(x)))
            sigma = jnp.clip(sigma,min_denom, None)

            x = mu+sigma*jx.random.normal(key,mu.shape)
            return x, {'mu':mu, 'sigma':sigma}
        elif(self.latent_type=='tanh_gaussian'):
            mu = hk.Linear(self.num_features)(hk.Flatten()(x))
            sigma = jx.nn.softplus(hk.Linear(self.num_features)(hk.Flatten()(x)))
            sigma = jnp.clip(sigma,min_denom, None)

            x = jx.nn.tanh(mu+sigma*jx.random.normal(key,mu.shape))
            return x, {'mu':mu, 'sigma':sigma}
        elif(self.latent_type=='categorical'):
            logits = jnp.reshape(hk.Linear(self.num_features*self.feature_width)(hk.Flatten()(x)), [-1, self.num_features,self.feature_width])

            probs = jx.nn.softmax(logits)
            log_probs = jx.nn.log_softmax(logits)

            x = jx.nn.one_hot(jx.random.categorical(key, logits),self.feature_width, axis=2)
            x = probs+jx.lax.stop_gradient(x-probs)
            return x, {'probs':probs, 'log_probs':log_probs}

class phi_conv_network(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_features = config['num_features']
        self.num_hidden_layers = config['num_hidden_layers']
        self.feature_width = config['feature_width']
        self.conv_depth = config['conv_depth']
        self.num_conv_filters = config['num_conv_filters']
        self.num_hidden_units = config['num_hidden_units']
        self.activation_function = activation_dict[config['activation']]
        self.latent_type = config['latent_type']

    def __call__(self, s, h, key):
        #encode image
        x = s
        for i in range(self.conv_depth):
            x = self.activation_function(hk.Conv2D(self.num_conv_filters*(2**i), 3, padding='VALID')(x))

        #combine image and recurrent state
        x = jnp.concatenate([h,hk.Flatten()(x)],axis=1)

        #pass both through a MLP
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))

        if(self.latent_type=='gaussian'):
            mu = hk.Linear(self.num_features)(hk.Flatten()(x))
            sigma = jx.nn.softplus(hk.Linear(self.num_features)(hk.Flatten()(x)))
            sigma = jnp.clip(sigma,min_denom, None)

            x = mu+sigma*jx.random.normal(key,mu.shape)
            return x, {'mu':mu, 'sigma':sigma}
        elif(self.latent_type=='tanh_gaussian'):
            mu = hk.Linear(self.num_features)(hk.Flatten()(x))
            sigma = jx.nn.softplus(hk.Linear(self.num_features)(hk.Flatten()(x)))
            sigma = jnp.clip(sigma,min_denom, None)

            x = jx.nn.tanh(mu+sigma*jx.random.normal(key,mu.shape))
            return x, {'mu':mu, 'sigma':sigma}
        elif(self.latent_type=='categorical'):
            logits = jnp.reshape(hk.Linear(self.num_features*self.feature_width)(hk.Flatten()(x)), [-1, self.num_features,self.feature_width])

            probs = jx.nn.softmax(logits)
            log_probs = jx.nn.log_softmax(logits)

            x = jx.nn.one_hot(jx.random.categorical(key, logits),self.feature_width, axis=2)
            x = probs+jx.lax.stop_gradient(x-probs)
            return x, {'probs':probs, 'log_probs':log_probs}

class phi_flat_network(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_features = config['num_features']
        self.feature_width = config['feature_width']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_hidden_units = config['num_hidden_units']
        self.activation_function = activation_dict[config['activation']]
        self.latent_type = config['latent_type']

    def __call__(self, s, h, key):
        x = jnp.concatenate([h,hk.Flatten()(s)],axis=1)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))

        if(self.latent_type=='gaussian'):
            mu = hk.Linear(self.num_features)(hk.Flatten()(x))
            sigma = jx.nn.softplus(hk.Linear(self.num_features)(hk.Flatten()(x)))
            sigma = jnp.clip(sigma,min_denom, None)

            x = mu+sigma*jx.random.normal(key,mu.shape)
            return x, {'mu':mu, 'sigma':sigma}
        elif(self.latent_type=='tanh_gaussian'):
            mu = hk.Linear(self.num_features)(hk.Flatten()(x))
            sigma = jx.nn.softplus(hk.Linear(self.num_features)(hk.Flatten()(x)))
            sigma = jnp.clip(sigma,min_denom, None)

            x = jx.nn.tanh(mu+sigma*jx.random.normal(key,mu.shape))
            return x, {'mu':mu, 'sigma':sigma}
        elif(self.latent_type=='categorical'):
            logits = jnp.reshape(hk.Linear(self.num_features*self.feature_width)(hk.Flatten()(x)), [-1, self.num_features,self.feature_width])

            probs = jx.nn.softmax(logits)
            log_probs = jx.nn.log_softmax(logits)

            x = jx.nn.one_hot(jx.random.categorical(key, logits),self.feature_width, axis=2)
            x = probs+jx.lax.stop_gradient(x-probs)
            return x, {'probs':probs, 'log_probs':log_probs}

class state_conv_network(hk.Module):
    def __init__(self, config, binary, state_shape, name=None):
        super().__init__(name=name)
        self.num_features = config['num_features']
        self.feature_width = config['feature_width']
        self.conv_depth = config['conv_depth']
        self.num_conv_filters = config['num_conv_filters']
        self.num_hidden_units = config['num_hidden_units']
        self.activation_function = activation_dict[config['activation']]
        self.binary = binary
        self.state_shape = state_shape

    def __call__(self, phi, h):
        desired_shape = [self.state_shape[0]-2*self.conv_depth,self.state_shape[1]-2*self.conv_depth,self.num_conv_filters*(2**(self.conv_depth-1))]
        num_units = 1
        for j in desired_shape:
            num_units*=j

        x = jnp.concatenate([h,hk.Flatten()(phi)],axis=1)
        # for i in range(num_hidden_layers-1):
        #     x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        # x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        x = self.activation_function(hk.Linear(num_units)(x))
        x = jnp.reshape(x, [-1]+desired_shape)
        for i in range(self.conv_depth-1):
            x = self.activation_function(hk.Conv2DTranspose(self.num_conv_filters*2**(self.conv_depth-i-1), 3, padding='VALID')(x))

        if(self.binary):
            logit = hk.Conv2DTranspose(self.state_shape[2], 3, output_shape=self.state_shape[:2], padding='VALID')(x)
            #Note this returns the logit of S, we wish to apply a sigmoid after to keep it bounded
            return {'logit':logit}
        else:
            mu = hk.Conv2DTranspose(self.state_shape[2], 3, output_shape=self.state_shape[:2], padding='VALID')(x)
            log_sigma = hk.Conv2DTranspose(self.state_shape[2], 3, output_shape=self.state_shape[:2], padding='VALID')(x)
            sigma = jx.nn.softplus(log_sigma)
            sigma = jnp.clip(sigma, min_denom, None)
            return {'mu':mu, 'sigma':sigma}

class state_flat_network(hk.Module):
    def __init__(self, config, binary, state_width, name=None):
        super().__init__(name=name)
        self.num_features = config['num_features']
        self.feature_width = config['feature_width']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_hidden_units = config['num_hidden_units']
        self.activation_function = activation_dict[config['activation']]
        self.binary = binary
        self.state_width = state_width

    def __call__(self, phi, h):
        x = jnp.concatenate([h,hk.Flatten()(phi)],axis=1)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        if(self.binary):
            logit = hk.Linear(self.state_width)(x)
            return {'logit':logit}
        else:
            mu = hk.Linear(self.state_width)(x)
            sigma = jx.nn.softplus(hk.Linear(self.state_width)(x))
            sigma = jnp.clip(sigma,min_denom, None)
            return {'mu':mu, 'sigma':sigma}

class critic_network(hk.Module):
    def __init__(self, config, name=None):
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_hidden_units = config['num_hidden_units']
        self.activation_function = activation_dict[config['activation']]
        super().__init__(name=name)

    def __call__(self, phi, h):
        x = jnp.concatenate([hk.Flatten()(phi), h],axis=1)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        V = hk.Linear(1)(x)[:,0]
        return V

class actor_network(hk.Module):
    def __init__(self, config, num_actions, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_hidden_units = config['num_hidden_units']
        self.num_actions = num_actions
        self.activation_function = activation_dict[config['activation']]

    def __call__(self, phi, h):
        x = jnp.concatenate([h,hk.Flatten()(phi), h],axis=1)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        pi_logit = hk.Linear(self.num_actions)(x)
        return pi_logit
