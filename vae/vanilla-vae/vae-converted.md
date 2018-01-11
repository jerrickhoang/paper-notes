# Variational Autoencoder (VAE)

by **Diederik P. Kingma** and **Max Welling**

https://arxiv.org/pdf/1312.6114.pdf

### Short summary

- The paper combines the traditional variational Bayes with neural networks. In particular, instead of the intractably approach by tradiational variational Bayes, the paper proposes an approach to estimate the variational lower bound using neural networks. In order for gradient descent to work, the paper introduce the **reparameterization trick** which pushes the stochastic operation of sampling out to the input, thus enabling the gradient to flow through the networks.


### Variational Autoencoder
Imagine you want to write a digit from 0-9 on a piece of paper. You first pick a pen, pick a digit then write it. The appearance of the digit on the paper depends on the kind of pen, the color of the ink, what digit you pick, etc. In generative models, this information is called latent variable.

We can state our generative problem as follows. Given a dataset <img src="svgs/93aa3b8a1a4411a0731ef8158cbf4b97.svg?invert_in_darkmode" align=middle width=98.85892499999999pt height=29.19113999999999pt/> are <img src="svgs/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.999985000000004pt height=22.46574pt/> i.i.d samples of a random variable <img src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908740000000003pt height=22.46574pt/> drawn from a hidden distribution. We further assume there is a latent variable <img src="svgs/5b51bd2e6f329245d425b8002d7cf942.svg?invert_in_darkmode" align=middle width=12.397275000000002pt height=22.46574pt/> and the hidden distribution is such that <img src="svgs/2a759819e95336580d8c1c33c72fd591.svg?invert_in_darkmode" align=middle width=184.55530499999998pt height=26.48447999999999pt/>. The problem is to draw more samples from this hidden distribution.

However, different tasks require a completely different structure of the latent space <img src="svgs/876325ffb025b2aa38e95162c10cb60b.svg?invert_in_darkmode" align=middle width=38.019465pt height=24.65759999999998pt/>. Generating a digit requires different latent information than generating a song. One interesting hypothesis is that for any latent distribution, there exists a deterministic transformation that maps a normal distribution to the latent one.

![normal](normal.png)

Based on this insight, we can somewhat safely start with <img src="svgs/663ed5415b381c36255beda9858be3e5.svg?invert_in_darkmode" align=middle width=112.67355pt height=24.65759999999998pt/>. Plugging this back to the first equation gives, <img src="svgs/3bc41882fbb7b2d22f3a7c597192638d.svg?invert_in_darkmode" align=middle width=273.925905pt height=26.48447999999999pt/>. One approach to solve this integral is using the Monte Carlo approximation,
\[
P(X) = \frac{1}{n}\sum_i P(X|Z=z_i)
\]
However, for high dimensional problem, n must be extremely larg for this approximationto be close to ground truth. The key idea in variational method is that for the most part <img src="svgs/e54f279f7436f43dc1dc8b47229f52f9.svg?invert_in_darkmode" align=middle width=57.494415000000004pt height=24.65759999999998pt/> will be nearly zero. So, we want to look for the "important" region in the space of <img src="svgs/5b51bd2e6f329245d425b8002d7cf942.svg?invert_in_darkmode" align=middle width=12.397275000000002pt height=22.46574pt/> by having another model <img src="svgs/e23062cf4b7fae37088e9b5b08214b2b.svg?invert_in_darkmode" align=middle width=65.09019pt height=24.65759999999998pt/> proposing the values of z that are likely to produce <img src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908740000000003pt height=22.46574pt/>,

\[
<p align="center"><img src="svgs/2c27abe484777349b7813dcb88910598.svg?invert_in_darkmode" align=middle width=605.7777pt height=88.74359999999999pt/></p>
\]
\[
<p align="center"><img src="svgs/1196c2914776d7a9b8567a755d195f93.svg?invert_in_darkmode" align=middle width=665.5704pt height=93.84804pt/></p>
\]

Therefore, <img src="svgs/f2d5dbecb29fd37e018fbf902243d1e4.svg?invert_in_darkmode" align=middle width=85.56042000000001pt height=24.65759999999998pt/> is the evidence lower bound (ELBO) of the marginal likelihood. Note, the variatonal lower bound can only be as good as our estimation of <img src="svgs/39f0b024f36d8c0bb0338437f6bcd0f0.svg?invert_in_darkmode" align=middle width=57.494415000000004pt height=24.65759999999998pt/> using <img src="svgs/e23062cf4b7fae37088e9b5b08214b2b.svg?invert_in_darkmode" align=middle width=65.09019pt height=24.65759999999998pt/>. In summary, to estimate the <img src="svgs/a8b02ba2136869c12cbfbc67d928ce05.svg?invert_in_darkmode" align=middle width=38.635905pt height=22.46574pt/>, we first need to estimate <img src="svgs/e23062cf4b7fae37088e9b5b08214b2b.svg?invert_in_darkmode" align=middle width=65.09019pt height=24.65759999999998pt/>, sample <img src="svgs/ea965f58a3c898813820f42fb8120c8e.svg?invert_in_darkmode" align=middle width=95.375445pt height=24.65759999999998pt/>, then estimate <img src="svgs/6a5441e4b83496a5083dcd7ecadbb3f2.svg?invert_in_darkmode" align=middle width=87.77967pt height=24.65759999999998pt/>. The first part is called the **encoder** which encodes the input into a hidden state. The second part is called the **decoder**, which given the hidden state, infers the input. The two parts combined is called [autoencoder](https://en.wikipedia.org/wiki/Autoencoder), (hence the name variational autoencoder).

However, in order to use learn encoder and decoder jointly using gradient based methods, all operations must be differentible. The sampling operation in the middle of the encoder and decoder is not. The paper introduces a technique called **reparameterization trick** which pushes the sampling operation to the input layer. It works by sampling <img src="svgs/8fa25af70d546a0a89e1ff6a464d46ae.svg?invert_in_darkmode" align=middle width=79.27623000000001pt height=24.65759999999998pt/>. If the **encoder** <img src="svgs/e23062cf4b7fae37088e9b5b08214b2b.svg?invert_in_darkmode" align=middle width=65.09019pt height=24.65759999999998pt/> produces <img src="svgs/b1b81553c3be1c02b95544f6e69405e9.svg?invert_in_darkmode" align=middle width=84.471255pt height=24.65759999999998pt/>, then the input to the **decoder** will then be <img src="svgs/76a4dc939c5b3c5918c3eb3d9aef992b.svg?invert_in_darkmode" align=middle width=170.401605pt height=29.19113999999999pt/>.

![reparam](reparam.png)

### Implementation

Let's look at an example using MNIST dataset.

- The encoder takes the 28x28 and produces a n-dimensional mean and variance for z (to ensure variance is positive we predict log <img src="svgs/813cd865c037c89fcdc609b25c465a05.svg?invert_in_darkmode" align=middle width=11.872245000000005pt height=22.46574pt/> instead, then <img src="svgs/74e08f9a03cebf55aaa690c4f22a7a0e.svg?invert_in_darkmode" align=middle width=34.576245pt height=27.91271999999999pt/> is positive )
```python
with tf.variable_scope('encoder'):
  fc = slim.fully_connected(self.input_x, 512, activation_fn=tf.nn.relu)
  fc = slim.fully_connected(fc, 384, activation_fn=tf.nn.relu)
  fc = slim.fully_connected(fc, 256, activation_fn=tf.nn.relu)
  mu = slim.fully_connected(fc, 10, activation_fn=None)
  log_sigma = slim.fully_connected(fc, 10, activation_fn=None)
```
- Reparameterization trick, first sample <img src="svgs/bd396e3fc49ce6aa671e37d9e2ef7e28.svg?invert_in_darkmode" align=middle width=81.326355pt height=24.65759999999998pt/>, then scale <img src="svgs/06e0d89d1ffdc346f9386f0e1725d30d.svg?invert_in_darkmode" align=middle width=170.401605pt height=29.19113999999999pt/>,
```python
with tf.variable_scope('z'):
  eps = tf.random_normal(shape=tf.shape(log_sigma), mean=0, stddev=1, dtype=tf.float32)
  self.z = mu + tf.sqrt(tf.exp(log_sigma)) * eps
```
- The decoder takes <img src="svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.367645000000003pt height=14.155350000000013pt/> then tries to reproduce input <img src="svgs/f84e86b97e20e45cc17d297dc794b3e8.svg?invert_in_darkmode" align=middle width=9.395100000000005pt height=22.831379999999992pt/>,
```python
with tf.variable_scope('decoder'):
  dec = slim.fully_connected(self.z, 256, activation_fn=tf.nn.relu)
  dec = slim.fully_connected(dec, 384, activation_fn=tf.nn.relu)
  dec = slim.fully_connected(dec, 512, activation_fn=tf.nn.relu)
  dec = slim.fully_connected(dec, 784, activation_fn=None)
```
- Our objective is to **maximize** the evidence lower bound (ELBO),
\[ \mathcal{L}(Q_{\theta}, X, Z) = E_{x \sim D}[E_{z\sim Q_{\theta(Z|X=x)}}[logP(X|Z=z)] - D_{KL}(Q_{\theta}(Z|X=x) || P(Z))] \]
So,
\[
<p align="center"><img src="svgs/e91d28c4311e54f0df7d536f33ba3e56.svg?invert_in_darkmode" align=middle width=579.1929pt height=134.118435pt/></p>
\]

- The implementation of the cross entropy can be as simple as,
```python
self.rec = tf.reduce_mean(tf.reduce_sum(
tf.nn.sigmoid_cross_entropy_with_logits(logits=dec, labels=self.input_x), 1))
```
- Since we assume Gaussian for <img src="svgs/876325ffb025b2aa38e95162c10cb60b.svg?invert_in_darkmode" align=middle width=38.019465pt height=24.65759999999998pt/> and <img src="svgs/07a1f332890382d9776c1b1e8f9d7b98.svg?invert_in_darkmode" align=middle width=38.178195pt height=24.65759999999998pt/>, the KL loss can be calculated in closed form,
\[
<p align="center"><img src="svgs/dd2fe0f4efeefa9ed80b65eaebcad489.svg?invert_in_darkmode" align=middle width=527.4159pt height=57.812205pt/></p>
\]

```python
self.kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(
tf.exp(log_sigma) + tf.square(mu) - 1. - log_sigma, 1))
```
- The total loss would just be the sum of the two losses
```python
 self.loss = tf.reduce_mean(self.rec + self.kl_loss)
 ```

 ### Results and Discussion
  ![reconstruct](reconstruct.png)
 - Here is the results of reconstruction for 100 random images after 10 minutes of training (10000 iterations, with batch size = 100, learning rate = 1e-3, dimension of z = 10). Assume the columns are 0-indexed, then the even columns contain reconstructed images and the odd columns contain real images.

 ![generated](generated.png)
 - Here are the digits generated by the same model from a gaussian latent space.
 - One interesting observation in training VAE is that: initially, both reconstruction loss and KL loss go down. At one point, KL loss starts going up while total loss still goes down. To understand this phenomenon, notice that in a correctly trained VAE and with structured data, KL loss cannot be 0. If KL loss equals 0, it means there is no structured subspace of the prior distribution for the latent variable. Or, the data is almost completely random. At first, both <img src="svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode" align=middle width=12.836835000000004pt height=22.46574pt/> and <img src="svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg?invert_in_darkmode" align=middle width=12.995565000000004pt height=22.46574pt/> start out to be random, so both losses go down as <img src="svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode" align=middle width=12.836835000000004pt height=22.46574pt/> gets good at producing the average of the data, and <img src="svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg?invert_in_darkmode" align=middle width=12.995565000000004pt height=22.46574pt/> gets good at finding the subspace in the latent space. After this phase, <img src="svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode" align=middle width=12.836835000000004pt height=22.46574pt/> and <img src="svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg?invert_in_darkmode" align=middle width=12.995565000000004pt height=22.46574pt/> start making trade-off. Specifically, if <img src="svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg?invert_in_darkmode" align=middle width=12.995565000000004pt height=22.46574pt/> allows more mass in the subspace, more noise will be generated by <img src="svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode" align=middle width=12.836835000000004pt height=22.46574pt/> hence the reconstruction loss will go up and vice versa.
 ![train_process](train_process.png)
 - To make things more interesting, we can train a different model with laten dimension = 2 and visualize it,
![top_infer](top_infer.png)
- If we sample spatially from the 2D gaussian of the latent space and with each value <img src="svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.367645000000003pt height=14.155350000000013pt/>, we generate a digit, then the top down view looks something like the picture above. We can see that the latent space encodes the skewness of the digits.
![top_label](top_label.png)
- If we repeat the same sampling mechanism and now encode the digits as colors, we can see that from the original Gaussian, the encoder networks has learnt a non-linear transformation that transoforms the Gaussian into different subspaces encoding different digits.

That concludes my summary and experiments with the original VAE, more theoretical analysis as well as experiments with recent variations to come.
