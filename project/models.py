from django.db import models

# Models

class Question(models.Model):
    question = models.CharField(max_length=2000)

    class Meta:
        verbose_name = 'Question'
        verbose_name_plural = 'Question'

    def __str__(self):
        return '{} - {}'.format(self.question)


class Answer(models.Model):
    answer = models.CharField(max_length=2000)

    class Meta:
        verbose_name = 'Answer'
        verbose_name_plural = 'Answer'

    def __str__(self):
        return '{} - {}'.format(self.answer)

class Context(models.Model):
    context = models.CharField(max_length=2000)

    class Meta:
        verbose_name = 'Context'
        verbose_name_plural = 'Context'

    def __str__(self):
        return '{} - {}'.format(self.context)
   


