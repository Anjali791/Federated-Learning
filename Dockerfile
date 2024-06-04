# start by pulling the python image
FROM python:3

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

# Step 6: Expose the port Flask is running on
EXPOSE 8001

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["app.py", "--host=0.0.0.0", "--port=8001" ]