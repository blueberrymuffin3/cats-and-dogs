<template lang="pug">
  .text-xs-center
    input(type="file" :disabled="isLoading" @change="fileInputChange" ref="fileInput")

    VBtn(:loading="isLoading" :disabled="!isValidFile" @click="analyzeImage()").primary.mb-4 Analyze Image

    div
      canvas(width=100 height=0 ref="imagePreview")
    template(v-if="showOutput")
      .headline(v-if="imageValue>0.5") {{(imageValue*100).toFixed(3)}}% Dog
      .headline(v-else) {{((1-imageValue)*100).toFixed(3)}}% Cat

      VProgressLinear(
        :size="150" :width="25" :rotate="90"
        color="orange" background-color="blue"
        :value="imageValue*100"
    )
</template>

<script>
import * as tf from "@tensorflow/tfjs";

export default {
  data: () => ({
    a: null,
    modelLoading: true,
    model: {},
    isValidFile: false,
    showOutput: false,
    isLoading: false,
    imageValue: NaN,
    imageData: null
  }),
  methods: {
    async loadModel(){
      this.isLoading = true
      this.model = await tf.loadLayersModel('/model/model.json')
      this.isLoading = false
    },
    async downloadImage(){
      // Clear the canvas
      this.$refs.imagePreview.height = 0
      
      let imageURL = URL.createObjectURL(this.$refs.fileInput.files[0])
      let imageData = new Image()
      imageData.src = imageURL
      await imageData.decode()

      imageData = await tf.browser.fromPixels(imageData)
      imageData = tf.image.resizeBilinear(imageData, [100, 100], true)
      tf.browser.toPixels(imageData, this.$refs.imagePreview)
      this.imageData = imageData
    },
    async analyzeImage(){
      this.isLoading = true
      await new Promise((res) => setTimeout(res))

      let imageData = this.imageData.toFloat().div(tf.scalar(127)).sub(tf.scalar(1)).reshape([1, 100, 100, 3])
      let prediction = await this.model.predict(imageData).data()
      
      this.imageValue = prediction[0]
      this.showOutput = true
      this.isLoading = false
    },
    async fileInputChange(){
      let fileExists = this.$refs.fileInput.files.length == 1
      this.showOutput = false
      if(fileExists){
        try{
          this.isLoading = true
          await this.downloadImage()
          this.isValidFile = true
        } catch(e) {
          // TODO: Should error be shown to user?
          this.isValidFile = false
        } finally {
          this.isLoading = false
        }
      } else {
        this.isValidFile = false
      }
    }
  },
  async created() {
    await this.loadModel()
  }
};
</script>

<style>
</style>
