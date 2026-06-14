#!/usr/bin/env python3
""" Module: 0-simple_gan """

import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
	"""Simple GAN model with custom training step."""

	def __init__(self, generator, discriminator, latent_generator,
				 real_examples, batch_size=200, disc_iter=2,
				 learning_rate=.005):
		super().__init__()
		self.latent_generator = latent_generator
		self.real_examples = real_examples
		self.generator = generator
		self.discriminator = discriminator
		self.batch_size = batch_size
		self.disc_iter = disc_iter

		self.learning_rate = learning_rate
		self.beta_1 = .5
		self.beta_2 = .9

		self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
		self.generator.optimizer = keras.optimizers.Adam(
			learning_rate=self.learning_rate,
			beta_1=self.beta_1,
			beta_2=self.beta_2,
		)
		self.generator.compile(
			optimizer=self.generator.optimizer,
			loss=self.generator.loss,
		)

		self.discriminator.loss = lambda x, y: (
			tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape))
			+ tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape))
		)
		self.discriminator.optimizer = keras.optimizers.Adam(
			learning_rate=self.learning_rate,
			beta_1=self.beta_1,
			beta_2=self.beta_2,
		)
		self.discriminator.compile(
			optimizer=self.discriminator.optimizer,
			loss=self.discriminator.loss,
		)

	def get_real_sample(self, size=None):
		"""Return a random batch of real examples."""
		if size is None:
			size = self.batch_size
		sorted_indices = tf.range(tf.shape(self.real_examples)[0])
		random_indices = tf.random.shuffle(sorted_indices)[:size]
		return tf.gather(self.real_examples, random_indices)

	def get_fake_sample(self, size=None, training=False):
		"""Return a batch of generated examples."""
		if size is None:
			size = self.batch_size
		latent = self.latent_generator(size)
		return self.generator(latent, training=training)

	def train_step(self, useless_argument):
		"""Perform one GAN training step."""
		discr_loss = None

		for _ in range(self.disc_iter):
			real_sample = self.get_real_sample()
			fake_sample = tf.stop_gradient(self.get_fake_sample(training=True))

			with tf.GradientTape() as tape:
				real_pred = self.discriminator(real_sample, training=True)
				fake_pred = self.discriminator(fake_sample, training=True)
				discr_loss = self.discriminator.loss(real_pred, fake_pred)

			gradients = tape.gradient(
				discr_loss, self.discriminator.trainable_variables
			)
			self.discriminator.optimizer.apply_gradients(
				zip(gradients, self.discriminator.trainable_variables)
			)

		with tf.GradientTape() as tape:
			fake_sample = self.get_fake_sample(training=True)
			fake_pred = self.discriminator(fake_sample, training=False)
			gen_loss = self.generator.loss(fake_pred)

		gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
		self.generator.optimizer.apply_gradients(
			zip(gradients, self.generator.trainable_variables)
		)

		return {"discr_loss": discr_loss, "gen_loss": gen_loss}

