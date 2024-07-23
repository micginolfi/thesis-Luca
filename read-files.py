#%%
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

#%%
#load the data
hdulist = pyfits.open('GALAXY_STARFORMING.fits')

header = hdulist[1].header
print(header.keys)

data = hdulist[1].data

# %%
# extract the redshift from the data
redshift = data['Z']

# extract the spectra from the data
spectra = data['int_flux']

#%%
# Plot a random spectrum
# run this cell multiple times to see different random spectra
random_id = np.random.randint(len(spectra))

# create a x-axis for the spectrum, sampling from 4000 to 9000 amstrongs
wavelength = np.linspace(4000, 9000, len(spectra[random_id]))

plt.plot(wavelength, spectra[random_id])
plt.xlabel('Wavelength [Angstroms]')
plt.ylabel('Flux')
plt.title('Spectrum of a Random Galaxy')
plt.show()

#%%
from sklearn.model_selection import train_test_split
# Split the data into train, validation, and test sets

X_train, X_test, y_train, y_test = train_test_split(spectra, redshift, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#%%
# Normalize the spectra

max_value = np.max(X_train)

X_train = X_train / max_value
X_val = X_val / max_value
X_test = X_test / max_value
